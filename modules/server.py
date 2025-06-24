import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from flask import Flask, request, jsonify
import time
from accelerate import Accelerator
import os
import uuid
import re  # Add regex support
from datetime import datetime
from io import BytesIO
import base64
import fitz  # PyMuPDF
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
import threading
import queue
import json
import re
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import html
import concurrent.futures  # For parallel processing

# Import modules
from modules.models import initialize_models
from modules.job_processing import job_queue, job_results, start_processing_thread

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

# Model paths and configuration
base_model_id = "./Ministral-8B-Instruct-2410"  # Local model path
embed_id = "maidalun1020/bce-embedding-base_v1"

# Initialize models
print(f"Initializing models with optimized settings for faster processing...")
embedding, text_generation_pipeline = initialize_models(base_model_id, embed_id)

# Start the job processing thread
processing_thread = start_processing_thread(embedding, text_generation_pipeline)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Model server is running"}), 200

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
            'message': 'Job queued for processing',
            'created_at': datetime.now().isoformat()
        }
        
        # Add job to the queue
        job_queue.put((job_id, instruction, file_path, original_filename))
        
        # Return the job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Document uploaded and queued for processing'
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
        response['pdf_base64'] = job['pdf_base64']
        response['response'] = job['response']
    
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
        'response': job['response'],
        'pdf_base64': job['pdf_base64']
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
            'message': 'Job queued for processing',
            'created_at': datetime.now().isoformat()
        }
        
        # Add job to the queue
        job_queue.put((job_id, instruction, file_path, temp_filename))
        
        # Return the job ID immediately
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Document queued for processing'
        }), 202
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
