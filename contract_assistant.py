import os
import time
from datetime import datetime
import uuid
import requests
from io import BytesIO
import base64
import json
from flask import session

# Handle PyMuPDF import with fallback options
try:
    import fitz
except ImportError:
    try:
        import PyMuPDF
        fitz = PyMuPDF
    except ImportError:
        raise ImportError("Could not import PyMuPDF. Please install with: pip install PyMuPDF==1.16.14")

from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors

from .document_storage import DocumentStorage

# Initialize paths using script location
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(script_dir, "..", "uploaded_docs")
PDF_OUTPUT_FOLDER = os.path.join(script_dir, "..", "generated_pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

# Model server configuration
MODEL_SERVER_URL = "http://172.31.28.218:5001"

def process_document(instruction, file):
    """Process document using the legacy model server"""
    start_time = time.time()
    try:
        print("\n=== Starting new query processing ===")
        
        # Initialize document storage
        doc_storage = DocumentStorage()
        
        # Save the document first - use anonymous if no session
        try:
            user_id = session.get('user_id', 'anonymous')
        except RuntimeError:
            user_id = 'anonymous'  # Fallback if session is not available
            
        file_info = doc_storage.save_document(file, 'contract_documents', user_id)
        file_path = file_info['file_path']
        
        # Upload document to the model server
        upload_start = time.time()
        print(f"Uploading document to server: {file.filename}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file.filename, f.read())}
            data = {'instruction': instruction}
            response = requests.post(
                f"{MODEL_SERVER_URL}/upload",
                files=files,
                data=data
            )
        
        # Check if upload was successful
        if response.status_code != 202:
            error_msg = response.json().get('error', 'Unknown error') if response.content else response.text
            print(f"Upload error (Status {response.status_code}): {error_msg}")
            raise Exception(f"Upload error: {error_msg}")
            
        # Get job ID from response
        job_data = response.json()
        job_id = job_data.get('job_id')
        
        if not job_id:
            raise Exception("No job ID received from server")
            
        print(f"File uploaded successfully. Job ID: {job_id}")
        print(f"Upload time: {time.time() - upload_start:.2f} seconds")
        
        # Poll for job status until complete
        polling_start = time.time()
        print(f"Polling for job status...")
        
        # Initialize status tracking
        status = "queued"
        progress = 0
        message = "Job queued for processing"
        max_poll_time = 1800  # 30 minutes maximum wait time
        poll_interval = 2     # Start with 2 second polling interval
        max_poll_interval = 15  # Maximum polling interval in seconds
        
        while status != "completed" and status != "error":
            # Check if we've exceeded maximum wait time
            if time.time() - polling_start > max_poll_time:
                raise Exception(f"Job timed out after {max_poll_time} seconds")
                
            # Wait before polling again
            time.sleep(poll_interval)
            
            # Gradually increase polling interval (up to max_poll_interval)
            poll_interval = min(poll_interval * 1.2, max_poll_interval)
            
            # Poll for job status
            status_response = requests.get(f"{MODEL_SERVER_URL}/job_status/{job_id}")
            
            if status_response.status_code != 200:
                error_msg = status_response.json().get('error', 'Unknown error') if status_response.content else status_response.text
                print(f"Status check error (Status {status_response.status_code}): {error_msg}")
                continue  # Try again after delay
                
            status_data = status_response.json()
    
            # Update tracking variables
            new_status = status_data.get('status', status)
            new_progress = status_data.get('progress', progress)
            new_message = status_data.get('message', message)
            
            # Only print update if there's a change
            if new_status != status or new_progress != progress or new_message != message:
                status = new_status
                progress = new_progress
                message = new_message
                print(f"Status: {status} | Progress: {progress}% | {message}")
            
            # Check if job is completed in this poll
            if status == "completed" or status == "error":
                    break
                    
        if status == "error":
            raise Exception(f"Job failed: {message}")
            
        print(f"Job completed. Polling time: {time.time() - polling_start:.2f} seconds")
                    
        # Get the full results
        print("Retrieving full results...")
        result_response = requests.get(f"{MODEL_SERVER_URL}/job_result/{job_id}")
        
        if result_response.status_code != 200:
            error_msg = result_response.json().get('error', 'Unknown error') if result_response.content else result_response.text
            print(f"Result retrieval error (Status {result_response.status_code}): {error_msg}")
            raise Exception(f"Result retrieval error: {error_msg}")
            
        result_data = result_response.json()
        
        # Extract the response and PDF data
        combined_response = result_data.get('response', '')
        pdf_base64 = result_data.get('pdf_base64', '')
        
        if not pdf_base64:
            raise Exception("No PDF content received from server")
                
        # Convert base64 to PDF buffer
        pdf_buffer = BytesIO(base64.b64decode(pdf_base64))
        
        # Save PDF locally
        pdf_path = save_pdf(pdf_buffer, file.filename)
        
        result = {
            'response': combined_response,
            'pdf_base64': pdf_base64,
            'pdf_path': pdf_path
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file_path}: {str(e)}")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        return result

    except Exception as e:
        error_time = time.time() - start_time
        error_msg = f"Error occurred after {error_time:.2f} seconds: {str(e)}"
        print(error_msg)
        return {'error': True, 'message': error_msg}


def save_pdf(pdf_buffer, original_filename):
    """Save PDF buffer to file and return path"""
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    pdf_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}_{unique_id}.pdf"
    pdf_path = os.path.join(PDF_OUTPUT_FOLDER, pdf_filename)
    
    # Save PDF
    with open(pdf_path, 'wb') as f:
        f.write(pdf_buffer.getvalue())
    
    print(f"[PDF Generated] Saved to: {pdf_path}")
    return pdf_path
