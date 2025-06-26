import time
from io import BytesIO
import base64
import os
import fitz
import html
import concurrent.futures
from bs4 import BeautifulSoup

from .client import invoke_mistral_model

# Extract text from PDF and convert to HTML to preserve formatting
def extract_text_as_html(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        html_output = ['<!DOCTYPE html><html><head><style>',
                      'body { font-family: Times New Roman, serif; font-size: 11pt; line-height: 1.5; }',
                      '</style></head><body>']
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("html")
            html_output.append(text)
        
        html_output.append('</body></html>')
        doc.close()
        
        return '\n'.join(html_output)
    except Exception as e:
        print(f"Error extracting HTML from PDF: {str(e)}")
        # Fallback to simple text extraction
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            for page_num in range(len(doc)):
                text_content += doc[page_num].get_text()
            doc.close()
            
            # Convert plain text to simple HTML
            html_content = f'<!DOCTYPE html><html><body><pre>{html.escape(text_content)}</pre></body></html>'
            return html_content
        except Exception as e2:
            print(f"Fallback extraction also failed: {str(e2)}")
            return ""

def prepare_mistral_prompt(html_content, instruction):
    """Create properly formatted prompt for Mistral model"""
    mistral_prompt = f"<s>[INST] "
    mistral_prompt += """You are a precise contract editor that strictly follows user instructions for modifying legal documents. 
Core requirements:
1. ONLY modify text explicitly mentioned in the user's instructions
2. NEVER add new content, clauses, or explanatory text not present in the original document
3. NEVER generate or invent new legal language
4. Preserve ALL original formatting, structure, and unmodified sections exactly as they appear
5. Return ONLY the edited text without any commentary, explanations, or additional content
6. If an instruction is unclear or cannot be executed precisely, return the original text unchanged
7. Maintain exact terminology from the original document - do not substitute or paraphrase terms
8. Process each section independently - do not cross-reference or combine content from different parts
9. PRECISELY maintain ALL formatting elements of the contract:
   - Preserve ALL HTML tags and structure exactly as they appear
   - Do not add, remove, or modify any HTML tags
   - Only change the text content within HTML elements as instructed
10. Return the complete HTML with its structure intact - do NOT normalize or change the layout
"""
    mistral_prompt += f"""Original HTML: {html_content}
---
Instruction: {instruction}

Return only the modified HTML without any additional commentary."""
    mistral_prompt += " [/INST]"
    
    # Debug output
    print(f"Prompt length: {len(mistral_prompt)} characters")
    prompt_sample = mistral_prompt[:500] + "..." if len(mistral_prompt) > 500 else mistral_prompt
    print(f"Prompt sample: {prompt_sample}")
    
    return mistral_prompt

def chunk_html_document(html_content, max_chunk_size=15000):
    """Split HTML document into processable chunks while preserving structure"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # If the document is small enough, process it in one go
    if len(html_content) <= max_chunk_size:
        return [html_content]
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    # Get all immediate children of the body
    if not soup.body:
        # If no body, treat the whole content as one chunk
        return [html_content]
    
    for element in soup.body.children:
        element_str = str(element)
        element_size = len(element_str)
        
        # If element is too large, try to break it down further
        if element_size > max_chunk_size:
            # For large divs or sections, split by their children
            if hasattr(element, 'children'):
                for child in element.children:
                    child_str = str(child)
                    child_size = len(child_str)
                    
                    if current_size + child_size > max_chunk_size and current_chunk:
                        # Wrap current chunk in basic HTML structure
                        wrapped_chunk = f"<!DOCTYPE html><html><head></head><body>{current_chunk}</body></html>"
                        chunks.append(wrapped_chunk)
                        current_chunk = child_str
                        current_size = child_size
                    else:
                        current_chunk += child_str
                        current_size += child_size
            else:
                # If we can't break it down further, add as a single chunk
                if current_chunk:
                    wrapped_chunk = f"<!DOCTYPE html><html><head></head><body>{current_chunk}</body></html>"
                    chunks.append(wrapped_chunk)
                
                wrapped_chunk = f"<!DOCTYPE html><html><head></head><body>{element_str}</body></html>"
                chunks.append(wrapped_chunk)
                current_chunk = ""
                current_size = 0
        elif current_size + element_size > max_chunk_size and current_chunk:
            # Add current chunk and start a new one
            wrapped_chunk = f"<!DOCTYPE html><html><head></head><body>{current_chunk}</body></html>"
            chunks.append(wrapped_chunk)
            current_chunk = element_str
            current_size = element_size
        else:
            # Add to current chunk
            current_chunk += element_str
            current_size += element_size
    
    # Add the last chunk
    if current_chunk:
        wrapped_chunk = f"<!DOCTYPE html><html><head></head><body>{current_chunk}</body></html>"
        chunks.append(wrapped_chunk)
    
    return chunks

def combine_html_chunks(html_chunks, instruction=""):
    """Combine processed HTML chunks back into one document"""
    combined_content = ""
    
    for i, chunk in enumerate(html_chunks):
        # Skip empty chunks
        if not chunk or len(chunk.strip()) < 10:
            print(f"Warning: Chunk {i+1} appears to be empty or too small, skipping")
            continue
            
        # Debug output
        print(f"Processing chunk {i+1} for combination, length: {len(chunk)}")
        
        try:
            # Extract just the body content from each chunk
            soup = BeautifulSoup(chunk, 'html.parser')
            if soup.body:
                # Get all content from body, excluding the body tag itself
                body_content = ''.join(str(tag) for tag in soup.body.contents)
                combined_content += body_content
                print(f"Added {len(body_content)} bytes from chunk {i+1}")
            else:
                # If no body tag found, just add the content as-is
                print(f"No body tag in chunk {i+1}, adding raw content")
                combined_content += chunk
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            # Add the chunk as-is if parsing fails
            combined_content += chunk
    
    # Check if we have any content
    if not combined_content or len(combined_content.strip()) < 50:
        print("Warning: Combined HTML content is empty or too small")
        combined_content = f"<div><h2>Processing Results</h2><p>The document was processed with the instruction: {html.escape(str(instruction))}</p></div>"
    
    # Wrap in complete HTML structure
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; font-size: 12pt; line-height: 1.5; margin: 1in; }}
        h1, h2, h3 {{ color: #333; margin-top: 1em; margin-bottom: 0.5em; }}
        p {{ margin-bottom: 0.5em; }}
    </style>
</head>
<body>
{combined_content}
</body>
</html>"""

def process_with_bedrock(instruction, file_path, original_filename):
    """Process document using AWS Bedrock"""
    start_time = time.time()
    try:
        print("\n=== Starting new Bedrock processing job ===")
        
        # Extract text as HTML to preserve formatting
        html_content = extract_text_as_html(file_path)
        
        # Check if HTML content was properly extracted
        if not html_content or len(html_content.strip()) < 100:
            print(f"Warning: HTML extraction produced insufficient content. Length: {len(html_content)}")
            # Try simple text extraction as fallback
            try:
                doc = fitz.open(file_path)
                text_content = ""
                for page_num in range(len(doc)):
                    text_content += doc[page_num].get_text()
                doc.close()
                
                # Convert text to basic HTML
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; font-size: 12pt; line-height: 1.5; }}
    </style>
</head>
<body>
    <pre>{html.escape(text_content)}</pre>
</body>
</html>"""
                print("Used fallback text extraction")
            except Exception as e:
                print(f"Fallback extraction also failed: {str(e)}")
        
        # Chunk the document
        chunks = chunk_html_document(html_content)
        total_chunks = len(chunks)
        print(f"Document split into {total_chunks} chunks for processing")
        
        # Process chunks with Bedrock in parallel
        processed_chunks = [None] * total_chunks
        
        def process_chunk_worker(args):
            chunk, idx = args
            print(f"Processing chunk {idx+1}/{total_chunks}")
            
            # Create prompt for this chunk
            prompt = prepare_mistral_prompt(chunk, instruction)
            
            # Process with Bedrock
            model_id = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-8x7b-instruct-v0:1')
            result = invoke_mistral_model(prompt, modelId=model_id)
            
            # Validate the result - make sure we got a proper response
            if not result or len(result.strip()) < 10:
                print(f"Warning: Empty or too small response for chunk {idx+1}")
                # Use the original chunk if the model returned nothing useful
                result = chunk
                print(f"Using original chunk content as fallback")
            
            print(f"Completed chunk {idx+1}/{total_chunks}")
            
            # Debug output to check response content
            result_sample = result[:200] + "..." if len(result) > 200 else result
            print(f"Chunk {idx+1} result sample: {result_sample}")
            
            return idx, result
        
        # Use parallel processing for chunks with ThreadPoolExecutor
        max_workers = min(total_chunks, 5)  # Limit concurrency
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk_worker, (chunk, i)): i for i, chunk in enumerate(chunks)}
            
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                processed_chunks[idx] = result
                print(f"Progress: {len([c for c in processed_chunks if c is not None])}/{total_chunks} chunks completed")
        
        # Combine chunks
        combined_html = combine_html_chunks(processed_chunks, instruction)
        
        # Generate PDF from HTML
        from xhtml2pdf import pisa
        pdf_buffer = BytesIO()
        
        # Ensure HTML has proper structure and content
        if not combined_html or len(combined_html.strip()) < 50:
            print("Warning: HTML content appears to be empty or too small")
            # Create a simple fallback content to make sure PDF isn't empty
            combined_html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; font-size: 12pt; line-height: 1.5; }}
    </style>
</head>
<body>
    <h1>Contract Processing Result</h1>
    <p>The system has processed your document based on the instruction: "{instruction}"</p>
    <p>Please check the content below:</p>
    <div>{combined_html}</div>
</body>
</html>"""
        
        # Add debug logging
        print(f"HTML content length: {len(combined_html)}")
        
        # Create PDF with proper error handling
        pdf_conversion_status = pisa.CreatePDF(
            combined_html, 
            dest=pdf_buffer,
            debug=1  # Enable debug output
        )
        
        if pdf_conversion_status.err:
            print(f"PDF generation error: {pdf_conversion_status.err}")
            # Try fallback method
            try:
                # Import from parent package
                from nxtAppCore.pdf_utils import generate_fallback_pdf
                # Generate basic PDF with just text
                response_text = BeautifulSoup(combined_html, 'html.parser').get_text()
                generate_fallback_pdf(pdf_buffer, response_text, "fallback")
            except Exception as pdf_err:
                print(f"Fallback PDF generation also failed: {str(pdf_err)}")
        
        pdf_buffer.seek(0)
        
        # Check if PDF is valid and has content
        if pdf_buffer.getvalue() and len(pdf_buffer.getvalue()) > 100:
            print(f"Generated PDF size: {len(pdf_buffer.getvalue())} bytes")
        else:
            print("Warning: Generated PDF is empty or too small")
        
        # Convert PDF to base64 for response
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        
        # Generate PDF path (keeping format compatible with existing code)
        from datetime import datetime
        import uuid
        
        # Get PDF_OUTPUT_FOLDER from parent package
        from nxtAppCore.contract_assistant import PDF_OUTPUT_FOLDER
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        pdf_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}_{unique_id}.pdf"
        pdf_path = os.path.join(PDF_OUTPUT_FOLDER, pdf_filename)
        
        # Save PDF
        with open(pdf_path, 'wb') as f:
            f.write(pdf_buffer.getvalue())
        print(f"[PDF Generated] Saved to: {pdf_path}")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return {
            'response': combined_html,
            'pdf_base64': pdf_base64,
            'pdf_path': pdf_path
        }
    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"Error occurred after {error_time:.2f} seconds: {str(e)}"
        print(error_msg)
        return {'error': True, 'message': error_msg} 