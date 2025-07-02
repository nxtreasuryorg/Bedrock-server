# Bedrock Contract Processing Server

A production-ready contract processing system powered by AWS Bedrock and advanced PDF analysis, designed to intelligently modify contract documents while preserving their professional formatting and layout.

## 🌟 Overview

This system combines AWS Bedrock's AI capabilities with sophisticated PDF analysis to provide enterprise-grade contract processing. It automatically detects document complexity and applies the optimal processing strategy to maintain professional formatting while making precise content modifications.

## 🚀 Key Features

### ✨ **Intelligent Document Processing**
- **Multi-layer PDF Analysis**: Automatically detects tables, multi-column layouts, forms, and complex structures
- **Adaptive Processing**: Smart strategy selection based on document complexity
- **Enhanced Layout Preservation**: Maintains professional formatting in complex documents
- **Content Modification**: Precise AI-driven text changes while preserving structure

### 🧠 **AWS Bedrock Integration**
- **Mistral AI Models**: Leverages custom imported Mistral models via AWS Bedrock
- **Model Warmup**: Intelligent scheduling to prevent cold starts and reduce latency
- **Error Handling**: Comprehensive retry logic with fallback mechanisms
- **Cost Optimization**: Efficient API usage with rate limiting and batching

### 🏗️ **Production Architecture**
- **Asynchronous Processing**: Queue-based job system for scalable document handling
- **RESTful API**: Clean endpoints for document upload, status tracking, and result retrieval
- **Deployment Ready**: Optimized for cloud deployment (Render, AWS, etc.)
- **In-Memory Processing**: No persistent file storage for enhanced security and performance

## 📋 Table of Contents

- [Architecture](#architecture)
- [Workflow & Pipeline](#workflow--pipeline)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Enhanced PDF Processing](#enhanced-pdf-processing)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## 🏛️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Flask Server   │───▶│  AWS Bedrock    │
│                 │    │                 │    │   (Mistral)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Job Processing │
                    │     Queue       │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Enhanced PDF  │
                    │    Analysis     │
                    └─────────────────┘
```

### Core Components

- **Flask Server** (`server.py`): Main API server with health checks and endpoints
- **Job Processing** (`modules/job_processing.py`): Asynchronous document processing engine
- **Bedrock Integration** (`modules/bedrock_integration.py`): AWS Bedrock client with retry logic
- **Enhanced PDF Utils** (`modules/pdf_utils.py`): Advanced document analysis and layout preservation
- **Warmup Scheduler** (`modules/warmup_scheduler.py`): Model availability management
- **Text Processing** (`modules/text_processing.py`): Document chunking and analysis utilities

## 🔄 Workflow & Pipeline

### 1. **Document Upload & Analysis**
```
📄 PDF Upload → 🔍 Document Analysis → 📊 Complexity Scoring → 🎯 Strategy Selection
```

**Process:**
1. Document uploaded via `/upload` endpoint
2. Multi-layer analysis detects:
   - Tables and structured data
   - Multi-column layouts
   - Form fields and interactive elements
   - Font styles and formatting
   - Document complexity score
3. Processing strategy selected based on analysis

### 2. **Intelligent Processing Strategy**

```python
# Strategy Selection Logic
if complexity_level < 3 and layout_type == 'standard':
    # Simple documents - fast processing
    strategy = 'standard_extraction'
elif layout_type == 'table_heavy':
    # Preserve table structures
    strategy = 'table_focused_processing'
elif layout_type == 'multi_column':
    # Maintain column layouts
    strategy = 'column_aware_processing'
elif layout_type == 'form_based':
    # Preserve form fields
    strategy = 'form_preserving_processing'
else:
    # Complex mixed layouts
    strategy = 'hybrid_processing'
```

### 3. **Enhanced PDF Extraction**

```
📄 PDF Input → 🏗️ Structure Analysis → 🎨 HTML Generation → ✨ Enhanced CSS
```

**Features:**
- **Table Detection**: Converts PDF tables to structured HTML with proper styling
- **Multi-Column Support**: Preserves column layouts using CSS Grid/Flexbox
- **Font Analysis**: Captures typography, colors, and styling information
- **Form Recognition**: Identifies and preserves interactive elements

### 4. **AI-Powered Content Modification**

```
📝 Text Chunks → 🤖 AWS Bedrock → 📋 Instruction Processing → ✏️ Content Changes
```

**Process:**
1. Document split into intelligent chunks (25K characters with 5K overlap)
2. Entity detection identifies key terms from instructions
3. Parallel processing via AWS Bedrock Mistral models
4. Changes applied while preserving document structure

### 5. **Advanced HTML-to-PDF Generation**

```
🎨 Enhanced HTML → 🔄 Structure Preservation → 📄 Professional PDF
```

**Two-Tier Approach:**
- **Primary**: `xhtml2pdf` with enhanced CSS for complex layouts
- **Fallback**: `ReportLab` for reliable basic PDF generation

### 6. **Result Delivery**

```
📄 Generated PDF → 📦 Base64 Encoding → 🚀 API Response → 🧹 Cleanup
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Conda or virtual environment

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/your-org/bedrock-server.git
cd bedrock-server
```

2. **Environment Setup**
```bash
# Create conda environment
conda create -n contract python=3.9
conda activate contract

# Install dependencies
pip install -r requirements.txt
```

3. **AWS Configuration**
```bash
# Set environment variables
export AWS_ACCESS_KEY_ID='your_access_key'
export AWS_SECRET_ACCESS_KEY='your_secret_key'
export AWS_REGION='us-east-1'
export BEDROCK_MODEL_ID='your_model_id'
```

4. **Start Server**
```bash
python server.py
```

Server runs on `http://localhost:5001`

## 📡 API Documentation

### Core Endpoints

#### `POST /upload`
Upload document for processing
```bash
curl -X POST http://localhost:5001/upload \
  -F "file=@contract.pdf" \
  -F "instruction=Change company name from 'ABC Corp' to 'XYZ Inc'"
```

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "queued", 
  "message": "Document uploaded and queued for processing"
}
```

#### `GET /job_status/<job_id>`
Check processing status
```bash
curl http://localhost:5001/job_status/uuid-here
```

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "processing",
  "progress": 65,
  "message": "Processing document: 3/5 chunks completed"
}
```

#### `GET /job_result/<job_id>`
Get processed document
```bash
curl http://localhost:5001/job_result/uuid-here
```

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "response": "Modified document text...",
  "pdf_base64": "base64-encoded-pdf-data"
}
```

### Monitoring Endpoints

#### `GET /health`
Server health check
```json
{
  "status": "healthy",
  "message": "Model server is running with AWS Bedrock integration",
  "aws_region": "us-east-1",
  "warmup_scheduler": "active"
}
```

#### `GET /warmup/stats`
Warmup scheduler statistics
```json
{
  "warmup_scheduler": {
    "status": "running",
    "interval_minutes": 15,
    "stats": {
      "total_warmups": 48,
      "successful_warmups": 47,
      "success_rate": 97.9
    }
  }
}
```

## 🎨 Enhanced PDF Processing

### Document Complexity Detection

The system automatically analyzes documents and assigns complexity scores:

```python
# Complexity Factors
complexity_score = base_score + 
                  (column_count * 1) + 
                  (has_tables * 3) + 
                  (has_forms * 2) + 
                  (high_text_density * 2)
```

### Layout Types

1. **Standard** (`complexity < 3`): Simple single-column documents
2. **Table Heavy** (`has_tables = true`): Documents with structured data
3. **Multi-Column** (`columns > 1`): Newsletter/report style layouts  
4. **Form Based** (`has_forms = true`): Interactive documents with fields

### Enhanced CSS Features

```css
/* Advanced Layout Preservation */
.page-container { page-break-after: always; }
.extracted-table { border-collapse: collapse; width: 100%; }
.multi-column-2 { columns: 2; column-gap: 20px; }
.form-field { border: 1px solid #ccc; background: #fafafa; }

/* Print Optimization */
@media print {
  .page-container { page-break-after: always; }
  .no-break { page-break-inside: avoid; }
}
```

## ⚙️ Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key  
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=your_model_arn

# Server Configuration
FLASK_PORT=5001
MAX_CONTENT_LENGTH=104857600  # 100MB

# Processing Configuration
CHUNK_SIZE=25000
CHUNK_OVERLAP=5000
MAX_WORKERS=5
WARMUP_INTERVAL_MINUTES=15
```

### Model Configuration

```python
# Bedrock Model Settings
{
    "model_id": "arn:aws:bedrock:us-east-1:account:imported-model/model-id",
    "max_tokens": 4000,
    "temperature": 0.4,
    "top_p": 0.7,
    "top_k": 1
}
```

## 🚀 Deployment

### Render Deployment

1. **Connect Repository** to Render
2. **Environment Variables**: Set AWS credentials and configuration
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `python server.py`

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001

CMD ["python", "server.py"]
```

### AWS ECS/Fargate

Use the provided `docker-compose.yml` for container orchestration.

## 💡 Usage Examples

### Basic Contract Modification

```python
import requests

# Upload contract
response = requests.post('http://localhost:5001/upload', 
    files={'file': open('contract.pdf', 'rb')},
    data={'instruction': 'Change payment terms from Net 30 to Net 45'})

job_id = response.json()['job_id']

# Poll for completion
import time
while True:
    status = requests.get(f'http://localhost:5001/job_status/{job_id}').json()
    if status['status'] == 'completed':
        break
    time.sleep(2)

# Get result
result = requests.get(f'http://localhost:5001/job_result/{job_id}').json()
pdf_data = base64.b64decode(result['pdf_base64'])

with open('modified_contract.pdf', 'wb') as f:
    f.write(pdf_data)
```

### Complex Document Processing

```python
# Process multi-column document with tables
instruction = """
1. Change company name from 'Acme Corp' to 'Beta Industries'
2. Update all monetary amounts by adding 10%
3. Modify delivery terms from 5 days to 3 days
"""

response = requests.post('http://localhost:5001/upload',
    files={'file': open('complex_contract.pdf', 'rb')},
    data={'instruction': instruction})
```

### Client Integration

```python
from contract_assistant import process_document

result = process_document(
    instruction="Update vendor name to 'New Supplier LLC'",
    file=uploaded_file
)

if result.get('error'):
    print(f"Error: {result['message']}")
else:
    print(f"Processing complete. PDF saved to: {result['pdf_path']}")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Ready Error
```
ModelNotReadyException: The imported model is not ready
```
**Solution**: Wait for model warmup or use fallback model

#### 2. AWS Credentials Error
```
AWS Credentials Error: Invalid or missing credentials
```
**Solution**: Verify environment variables and IAM permissions

#### 3. PDF Generation Failed
```
PDF generation failed: HTML conversion error
```
**Solution**: Check document complexity and CSS compatibility

#### 4. Memory Issues
```
OutOfMemoryError during processing
```
**Solution**: Reduce chunk size or increase server memory

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Adjust chunk size** for your document types
2. **Tune worker count** based on server capacity  
3. **Monitor warmup efficiency** via `/warmup/stats`
4. **Use complexity scoring** to optimize processing strategy

## 📊 Monitoring & Metrics

### Key Metrics

- **Processing Time**: Average document processing duration
- **Success Rate**: Percentage of successful completions
- **Warmup Efficiency**: Model availability statistics
- **Error Distribution**: Common failure patterns

### Health Checks

- **Endpoint Monitoring**: `/health` for service availability
- **Model Status**: `/warmup/stats` for AI model health
- **Queue Status**: `/debug/queue` for processing backlog

## 🤝 Development

### Internal Development Workflow

1. Create feature branch (`git checkout -b feature/enhancement`)
2. Implement changes and test thoroughly
3. Update documentation as needed
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/enhancement`)
6. Create internal review request

## 📄 License

This project is for private/internal use only. All rights reserved.



---
