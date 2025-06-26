import time
import os
import uuid
from datetime import datetime
import threading
import queue
import base64
import html
from io import BytesIO
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import concurrent.futures  # Added for parallel processing
import re

from modules.pdf_utils import extract_text_as_html, process_html_with_model, generate_fallback_pdf, save_pdf
from modules.text_processing import (
    process_chunk_with_change_detection, 
    find_instruction_targets, 
    prioritize_chunks
)
from modules.bedrock_integration import BedrockClient

# Initialize Bedrock client
bedrock_client = BedrockClient()

# Function to extract key entities from instruction
def extract_key_entities(instruction):
    """Extract key entities like company names, addresses, etc. from instruction"""
    entities = []
    
    # Extract quoted text as potential entities
    quoted_entities = re.findall(r'[\'"]([^\'"]+)[\'"]', instruction)
    entities.extend([entity for entity in quoted_entities if len(entity) > 3])
    
    # Extract company name patterns
    company_patterns = [
        r'(?:Company|Provider|Client|Vendor|Contractor|Supplier|Customer)\s+Name.*?[\'"]([^\'"]+)[\'"]',
        r'[Cc]hange\s+(?:the\s+)?([^\'"].*?(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co))[^\'"].*?from',
        r'[Uu]pdate\s+(?:the\s+)?([^\'"].*?(?:Inc|LLC|Ltd|Corporation|Corp|Company|Co))[^\'"].*?from',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, instruction)
        entities.extend([match.strip() for match in matches if len(match.strip()) > 3])
    
    # Remove duplicates while preserving order
    unique_entities = []
    for entity in entities:
        if entity not in unique_entities:
            unique_entities.append(entity)
    
    return unique_entities

# Job processing queue and status tracking
job_queue = queue.Queue()
job_results = {}
processing_thread = None
should_process = True

# Process document and generate response
def process_document(job_id, instruction, file_path, original_filename, embedding=None, text_generation_pipeline=None):
    """Process document with AWS Bedrock - embedding and text_generation_pipeline parameters kept for compatibility but not used"""
    start_time = time.time()
    print(f"Starting job {job_id} with file: {original_filename}")
    
    try:
        # Update job status
        job_results[job_id]['status'] = 'processing'
        job_results[job_id]['progress'] = 10
        
        # Document content and chunking
        extract_start = time.time()
        document_content = ""
        html_content = ""
        
        if file_path.lower().endswith('.pdf'):
            try:
                # Extract as HTML to preserve formatting
                html_content = extract_text_as_html(file_path)
                
                # Also extract plain text for processing with the model
                doc = fitz.open(file_path)
                all_text = []
                for page_num, page in enumerate(doc):
                    all_text.append(page.get_text())
                document_content = "\n\n".join(all_text)
                print(f"Extracted text from {len(doc)} pages")
                doc.close()
                
                # Update job status
                job_results[job_id]['status'] = 'processing'
                job_results[job_id]['progress'] = 30
                job_results[job_id]['message'] = f"Extracted text from {len(all_text)} pages"
            except Exception as e:
                print(f"Error during PDF text extraction: {str(e)}")
                # Try simpler extraction
                try:
                    doc = fitz.open(file_path)
                    document_content = ""
                    for page_num in range(len(doc)):
                        document_content += doc[page_num].get_text()
                    doc.close()
                except Exception as e2:
                    print(f"Secondary PDF extraction also failed: {str(e2)}")
                    # If both methods fail, try to read as binary
                    with open(file_path, 'rb') as f:
                        document_content = f.read().decode('utf-8', errors='replace')
        else:
            # For non-PDF files, just read the content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                document_content = f.read()
            
            # Create simple HTML for non-PDF files
            html_content = f'<!DOCTYPE html><html><body><pre>{html.escape(document_content)}</pre></body></html>'
        
        print(f"Text extraction time: {time.time() - extract_start:.2f} seconds")

        # Update job status
        job_results[job_id]['status'] = 'processing'
        job_results[job_id]['progress'] = 40
        job_results[job_id]['message'] = "Text extracted, running inference..."

        # Process document content - we'll use the HTML for final output
        model_start = time.time()
        
        # Pre-process the instruction to identify potential targets
        target_sections = find_instruction_targets(instruction, document_content)
        
        # Extract and highlight key entities mentioned in the instruction
        entities_to_highlight = extract_key_entities(instruction)
        if entities_to_highlight:
            print(f"Identified {len(entities_to_highlight)} key entities to focus on: {', '.join(entities_to_highlight)}")
            # Pre-scan document for these entities
            found_entities = []
            for entity in entities_to_highlight:
                if entity.lower() in document_content.lower():
                    found_entities.append(entity)
                    print(f"Found entity '{entity}' in document")
                    
                    # Add entire paragraphs containing this entity to target sections
                    entity_lower = entity.lower()
                    doc_lower = document_content.lower()
                    entity_pos = doc_lower.find(entity_lower)
                    while entity_pos >= 0:
                        # Extract paragraph containing this entity
                        para_start = document_content.rfind('\n\n', 0, entity_pos)
                        if para_start == -1:
                            para_start = 0
                        else:
                            para_start += 2  # Skip the newlines
                            
                        para_end = document_content.find('\n\n', entity_pos)
                        if para_end == -1:
                            para_end = len(document_content)
                            
                        paragraph = document_content[para_start:para_end]
                        target_sections.append(paragraph)
                        
                        # Move to next occurrence
                        entity_pos = doc_lower.find(entity_lower, entity_pos + len(entity))
            
            if not found_entities:
                print(f"WARNING: Could not find these entities in the document: {', '.join(entities_to_highlight)}")
                
                # Try partial matching
                for entity in entities_to_highlight:
                    # Split into words and try to find longer parts
                    words = entity.split()
                    if len(words) >= 2:
                        for i in range(len(words) - 1):
                            partial = ' '.join(words[i:i+2])
                            if len(partial) >= 5 and partial.lower() in document_content.lower():
                                print(f"Found partial match for '{entity}': '{partial}'")
                                doc_lower = document_content.lower()
                                partial_lower = partial.lower()
                                partial_pos = doc_lower.find(partial_lower)
                                
                                # Extract surrounding context
                                context_start = max(0, partial_pos - 100)
                                context_end = min(len(document_content), partial_pos + len(partial) + 100)
                                context = document_content[context_start:context_end]
                                target_sections.append(context)
        
        # Smart chunking - use larger chunk size for faster processing, smaller overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=25000,  # Changed from 80000 to 25000 characters per chunk
            chunk_overlap=5000,  # Increased from 2000 to 5000 for better context preservation
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(document_content)
        
        # Try to process the entire document if it's small enough
        if len(document_content) < 25000:  # Updated threshold to match new chunk size
            print("Document small enough to process in one chunk - skipping chunking")
            # Use HTML content for processing if available to preserve formatting
            if html_content and len(html_content) > len(document_content):
                chunks = [html_content]
                print("Using HTML content for processing to preserve formatting")
            else:
                chunks = [document_content]
        elif len(chunks) <= 2:
            print(f"Document can be processed with {len(chunks)} chunks")
        else:
            # If we have identified target sections, prioritize chunks containing them
            if target_sections:
                print(f"Identified {len(target_sections)} potential target sections from instruction")
                chunks = prioritize_chunks(chunks, target_sections)
                print(f"Prioritized {len(chunks)} chunks based on relevance to instruction")
                
        # Process chunks with the model (plain text for now)
        processed_chunks = [None] * len(chunks)  # Pre-allocate result list
        total_chunks = len(chunks)
        changes_detected = False
        
        # Define a worker function for parallel processing with Bedrock
        def process_chunk_worker(args):
            chunk, idx = args
            chunk_id = f"{idx+1}/{total_chunks}"  # Format as "current/total"
            try:
                result, changed = process_chunk_with_change_detection(chunk, instruction, chunk_id)
                return idx, result, changed
            except Exception as e:
                error_msg = str(e)
                print(f"Error processing chunk {chunk_id}: {error_msg}")
                
                # Check for credential errors
                if "AWS Credentials Error" in error_msg or "security token" in error_msg.lower():
                    # This is a critical error that won't resolve with retries
                    raise Exception(f"AWS Bedrock credentials error: {error_msg}")
                
                # For other errors, return original chunk
                return idx, chunk, False
        
        # Use parallel processing for chunks with ThreadPoolExecutor
        # Add rate limiting for Bedrock API
        max_workers = min(total_chunks, 5)  # Limit to 5 concurrent requests to avoid rate limiting
        print(f"Processing {total_chunks} chunks with {max_workers} workers via AWS Bedrock")
        
        chunk_args = [(chunk, i) for i, chunk in enumerate(chunks)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk_worker, arg): arg for arg in chunk_args}
            completed = 0
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result, changed = future.result()
                    processed_chunks[idx] = result
                    completed += 1
                    
                    if changed:
                        changes_detected = True
                        print(f"Chunk {idx+1} was modified according to instruction")
                    
                    # Update progress
                    progress_pct = 40 + int(completed / total_chunks * 30)  # From 40% to 70%
                    job_results[job_id]['progress'] = progress_pct
                    job_results[job_id]['message'] = f"Processing document: {completed}/{total_chunks} chunks completed via AWS Bedrock"
                    print(f"Progress: {completed}/{total_chunks} chunks processed")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Critical error in chunk processing: {error_msg}")
                    
                    # Update job with error status
                    job_results[job_id]['status'] = 'error'
                    job_results[job_id]['message'] = f"Processing failed: {error_msg}"
                    job_results[job_id]['progress'] = 100
                    raise Exception(error_msg)
        
        # Combine chunks
        combined_response = "\n\n".join(processed_chunks)
        
        print(f"Model processing time: {time.time() - model_start:.2f} seconds")
        print(f"Combined response length: {len(combined_response)} characters")
        print(f"Combined response preview: {combined_response[:500]}...")
        
        if not changes_detected:
            print("WARNING: No changes detected in any chunk. The instruction may not match document content.")
        else:
            print(f"✅ Changes detected in document processing")
        
        # Update job status
        job_results[job_id]['status'] = 'processing'
        job_results[job_id]['progress'] = 70
        job_results[job_id]['message'] = "Inference complete, generating PDF..."

        # Process HTML with the model to maintain structure
        html_processing_start = time.time()
        
        # Process the HTML content with our changes
        processed_html = process_html_with_model(html_content, combined_response)
        
        print(f"HTML processing time: {time.time() - html_processing_start:.2f} seconds")
        
        # Generate PDF from the processed HTML
        pdf_start = time.time()
        pdf_buffer = BytesIO()
        
        try:
            # Convert the HTML to PDF using xhtml2pdf
            from xhtml2pdf import pisa
            pisa.CreatePDF(processed_html, dest=pdf_buffer)
            
            # Update job status
            job_results[job_id]['status'] = 'processing'
            job_results[job_id]['progress'] = 85
            job_results[job_id]['message'] = "Created professional contract document with preserved formatting"
                
        except Exception as e:
            print(f"Error during HTML-to-PDF generation: {str(e)}")
            # Create a simpler PDF with fallback method
            try:
                # Use the original code for PDF generation
                generate_fallback_pdf(pdf_buffer, combined_response, job_id)
                
                # Update job status
                job_results[job_id]['status'] = 'processing'
                job_results[job_id]['progress'] = 85
                job_results[job_id]['message'] = "Created basic contract document"
            except Exception as e2:
                print(f"Fallback PDF generation also failed: {str(e2)}")
                # Mark job as error
                job_results[job_id]['status'] = 'error'
                job_results[job_id]['message'] = f"PDF generation failed: {str(e2)}"
                job_results[job_id]['progress'] = 100
                raise Exception(f"PDF generation failed: {str(e2)}")
        
        print(f"PDF generation time: {time.time() - pdf_start:.2f} seconds")
        
        # Save PDF to file temporarily (only for generating base64)
        pdf_path = None
        try:
            pdf_path = save_pdf(pdf_buffer, original_filename)
            # Encode PDF as base64 for sending to client
            pdf_buffer.seek(0)
            pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
            
            # Update job with results (don't store pdf_path since we'll delete it)
            job_results[job_id]['status'] = 'completed'
            job_results[job_id]['progress'] = 100
            job_results[job_id]['message'] = "Processing complete"
            job_results[job_id]['response'] = combined_response
            job_results[job_id]['pdf_base64'] = pdf_base64
            # Don't store pdf_path since file will be deleted
            
        except Exception as e:
            error_msg = f"Error saving PDF: {str(e)}"
            print(error_msg)
            # Update job with error
            job_results[job_id]['status'] = 'error'
            job_results[job_id]['message'] = error_msg
            job_results[job_id]['progress'] = 100
            raise Exception(error_msg)
        finally:
            # Clean up all files - uploaded document and generated PDF
            files_to_clean = [file_path]
            if pdf_path:
                files_to_clean.append(pdf_path)
        
            for file_to_remove in files_to_clean:
                try:
                    if os.path.exists(file_to_remove):
                        os.remove(file_to_remove)
                        print(f"✅ Cleaned up file: {file_to_remove}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not remove file {file_to_remove}: {str(e)}")
        
        print(f"Job {job_id} completed successfully in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"Error occurred after {error_time:.2f} seconds: {str(e)}"
        print(error_msg)
        
        # Update job with error
        job_results[job_id]['status'] = 'error'
        job_results[job_id]['message'] = error_msg
        job_results[job_id]['progress'] = 100
        
        # Clean up uploaded file even on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Cleaned up uploaded file after error: {file_path}")
        except Exception as cleanup_e:
            print(f"⚠️ Warning: Could not remove uploaded file after error {file_path}: {str(cleanup_e)}")

# Job processing worker thread
def process_jobs(embedding=None, text_generation_pipeline=None):
    """Background thread for processing jobs from the queue - updated for Bedrock"""
    while should_process:
        try:
            # Get a job from the queue with a timeout
            try:
                job_id, instruction, file_path, original_filename = job_queue.get(timeout=1)
                # Process the job with Bedrock (ignoring embedding and text_generation_pipeline)
                process_document(job_id, instruction, file_path, original_filename)
            except queue.Empty:
                # No jobs to process, wait a bit
                time.sleep(0.1)
                continue
                
        except Exception as e:
            print(f"Error in job processing thread: {str(e)}")
            time.sleep(1)  # Wait a bit before trying again

# Start the job processing thread
def start_processing_thread(embedding=None, text_generation_pipeline=None):
    """Start the background thread for job processing - updated for Bedrock"""
    global processing_thread
    processing_thread = threading.Thread(
        target=process_jobs, 
        args=(),  # No need to pass models anymore
        daemon=True
    )
    processing_thread.start()
    return processing_thread 