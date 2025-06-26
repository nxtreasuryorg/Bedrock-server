import os
import uuid
import re
from datetime import datetime
from io import BytesIO
import base64
import fitz  # PyMuPDF
import threading
import queue
import json
import re
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import html
import concurrent.futures  # For parallel processing
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Import modules
from modules.job_processing import job_queue, job_results, start_processing_thread, process_document
from modules.bedrock_integration import BedrockClient
from modules.warmup_scheduler import initialize_warmup_scheduler, get_warmup_scheduler, update_last_request_time

# Initialize Bedrock client
bedrock_client = BedrockClient()

# Configure Flask with increased max content length (100MB)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['JSON_AS_ASCII'] = False  # Properly handle Unicode

# Initialize paths for document storage
script_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(script_dir, "uploaded_docs")
PDF_OUTPUT_FOLDER = os.path.join(script_dir, "generated_pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

# Print startup message
print(f"Starting server with AWS Bedrock integration")
print(f"AWS Region: {os.environ.get('AWS_REGION', 'Not set')}")
print(f"Upload folder: {UPLOAD_FOLDER}")
print(f"PDF output folder: {PDF_OUTPUT_FOLDER}")

# Start the job processing thread with Bedrock
processing_thread = start_processing_thread()
print(f"🔄 Processing thread started: {processing_thread is not None}")

# Initialize and start the warmup scheduler (15-minute intervals)
warmup_scheduler = initialize_warmup_scheduler(warmup_interval_minutes=15)
print(f"🔥 Warmup scheduler initialized - keeping model warm every 15 minutes")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Get warmup stats for health check
    scheduler = get_warmup_scheduler()
    warmup_status = "active" if scheduler and scheduler.is_running else "inactive"
    
    return jsonify({
        "status": "healthy", 
        "message": "Model server is running with AWS Bedrock integration",
        "aws_region": os.environ.get('AWS_REGION', 'Not set'),
        "warmup_scheduler": warmup_status
    }), 200

@app.route('/warmup/stats', methods=['GET'])
def warmup_stats():
    """Get warmup scheduler statistics"""
    scheduler = get_warmup_scheduler()
    if not scheduler:
        return jsonify({"error": "Warmup scheduler not initialized"}), 500
        
    stats = scheduler.get_stats()
    return jsonify({
        "warmup_scheduler": {
            "status": "running" if scheduler.is_running else "stopped",
            "interval_minutes": scheduler.warmup_interval / 60,
            "stats": stats
        }
    }), 200

@app.route('/warmup/trigger', methods=['POST'])
def trigger_warmup():
    """Manually trigger a warmup request (for testing)"""
    scheduler = get_warmup_scheduler()
    if not scheduler:
        return jsonify({"error": "Warmup scheduler not initialized"}), 500
        
    # Send warmup request
    success = scheduler.send_warmup_request()
    
    return jsonify({
        "message": "Manual warmup triggered",
        "success": success,
        "stats": scheduler.get_stats()
    }), 200 if success else 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload document for processing"""
    try:
        instruction = request.form.get('instruction')
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not instruction:
            return jsonify({"error": "No instruction provided"}), 400
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save the file
        original_filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{original_filename}")
        file.save(file_path)
        
        # Create job entry
        job_results[job_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued for processing with AWS Bedrock',
            'created_at': datetime.now().isoformat()
        }
        
        # Add job to the queue
        job_queue.put((job_id, instruction, file_path, original_filename))
        
        # Check if processing thread is alive - if not, trigger immediate processing
        thread_alive = processing_thread.is_alive() if processing_thread else False
        if not thread_alive:
            print(f"⚠️ Processing thread is dead, triggering immediate processing for job {job_id}")
            # Process in background thread to avoid blocking the upload response
            import threading
            
            def process_immediately():
                try:
                    # Get the job we just added
                    if not job_queue.empty():
                        job_data = job_queue.get(timeout=1)
                        job_id_from_queue, instruction_from_queue, file_path_from_queue, filename_from_queue = job_data
                        print(f"Processing job {job_id_from_queue} immediately")
                        process_document(job_id_from_queue, instruction_from_queue, file_path_from_queue, filename_from_queue)
                except Exception as e:
                    print(f"Immediate processing error for {job_id}: {e}")
                    if job_id in job_results:
                        job_results[job_id]['status'] = 'error'
                        job_results[job_id]['message'] = f"Processing error: {str(e)}"
            
            immediate_thread = threading.Thread(target=process_immediately, daemon=True)
            immediate_thread.start()
            
            job_results[job_id]['message'] = 'Processing started immediately (background thread backup)'
        
        # Update warmup scheduler - real user request received
        update_last_request_time()
        
        # Return the job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Document uploaded and queued for processing with AWS Bedrock'
        }), 202
        
    except Exception as e:
        print(f"Error handling document upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a job"""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
        
    job = job_results[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message']
    }
    
    # Include results if job is completed
    if job['status'] == 'completed':
        response['pdf_base64'] = job.get('pdf_base64', '')
        response['response'] = job.get('response', '')
    
    return jsonify(response), 200

@app.route('/job_result/<job_id>', methods=['GET'])
def get_job_result(job_id):
    """Get the full result of a completed job"""
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
        
    job = job_results[job_id]
    if job['status'] != 'completed':
        return jsonify({
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'message': 'Job not completed yet'
        }), 202
    
    # Return the full result
    return jsonify({
        'job_id': job_id,
        'status': 'completed',
        'response': job.get('response', ''),
        'pdf_base64': job.get('pdf_base64', '')
    }), 200

@app.route('/process_text', methods=['POST'])
def process_text():
    """Legacy endpoint that redirects to the new asynchronous flow"""
    try:
        data = request.get_json()
        
        if not data or 'instruction' not in data or 'document_content' not in data:
            return jsonify({"error": "Missing instruction or document content"}), 400
            
        instruction = data['instruction']
        document_content = data['document_content']
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save the document content to a temporary file
        file_ext = '.pdf' if document_content.startswith('%PDF') else '.txt'
        temp_filename = f"temp_doc_{job_id}{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        mode = 'wb' if file_ext == '.pdf' else 'w'
        with open(file_path, mode) as f:
            if file_ext == '.pdf':
                f.write(document_content.encode('utf-8', errors='replace'))
            else:
                f.write(document_content)
        
        # Create job entry
        job_results[job_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Job queued for processing with AWS Bedrock',
            'created_at': datetime.now().isoformat()
        }
        
        # Add job to the queue
        job_queue.put((job_id, instruction, file_path, temp_filename))
        
        # Check if processing thread is alive - if not, trigger immediate processing
        thread_alive = processing_thread.is_alive() if processing_thread else False
        if not thread_alive:
            print(f"⚠️ Processing thread is dead, triggering immediate processing for job {job_id}")
            # Process in background thread to avoid blocking the upload response
            import threading
            
            def process_immediately():
                try:
                    # Get the job we just added
                    if not job_queue.empty():
                        job_data = job_queue.get(timeout=1)
                        job_id_from_queue, instruction_from_queue, file_path_from_queue, filename_from_queue = job_data
                        print(f"Processing job {job_id_from_queue} immediately")
                        process_document(job_id_from_queue, instruction_from_queue, file_path_from_queue, filename_from_queue)
                except Exception as e:
                    print(f"Immediate processing error for {job_id}: {e}")
                    if job_id in job_results:
                        job_results[job_id]['status'] = 'error'
                        job_results[job_id]['message'] = f"Processing error: {str(e)}"
            
            immediate_thread = threading.Thread(target=process_immediately, daemon=True)
            immediate_thread.start()
            
            job_results[job_id]['message'] = 'Processing started immediately (background thread backup)'
        
        # Update warmup scheduler - real user request received
        update_last_request_time()
        
        # Return the job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Document queued for processing with AWS Bedrock'
        }), 202
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/queue', methods=['GET'])
def debug_queue():
    """Debug endpoint to check queue status and trigger processing"""
    queue_size = job_queue.qsize()
    thread_alive = processing_thread.is_alive() if processing_thread else False
    
    # Count jobs by status
    status_counts = {}
    for job_id, job in job_results.items():
        status = job['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return jsonify({
        "queue_size": queue_size,
        "processing_thread_alive": thread_alive,
        "job_status_counts": status_counts,
        "total_jobs": len(job_results)
    }), 200

@app.route('/debug/process_queue', methods=['POST'])
def trigger_queue_processing():
    """Manually trigger queue processing (for debugging Render issues)"""
    try:
        processed_count = 0
        max_jobs = 5  # Limit to prevent timeout
        
        while not job_queue.empty() and processed_count < max_jobs:
            try:
                # Get job from queue with short timeout
                job_id, instruction, file_path, original_filename = job_queue.get(timeout=1)
                
                # Process immediately in the current request context
                print(f"Manually processing job {job_id}")
                
                # Start processing in background thread to avoid blocking the request
                import threading
                
                def process_in_background():
                    try:
                        process_document(job_id, instruction, file_path, original_filename)
                    except Exception as e:
                        print(f"Background processing error for {job_id}: {e}")
                        if job_id in job_results:
                            job_results[job_id]['status'] = 'error'
                            job_results[job_id]['message'] = f"Processing error: {str(e)}"
                
                bg_thread = threading.Thread(target=process_in_background, daemon=True)
                bg_thread.start()
                
                processed_count += 1
                
            except queue.Empty:
                break
        
        return jsonify({
            "message": f"Triggered processing for {processed_count} jobs",
            "processed_jobs": processed_count,
            "remaining_queue_size": job_queue.qsize()
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Queue processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
