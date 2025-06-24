# Contract Assistant - AWS Bedrock Integration

This is an implementation of the Contract Assistant that uses AWS Bedrock for model inference instead of local model hosting.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements_bedrock.txt
   ```

2. Configure AWS credentials:
   - Edit `bedrock_integration.py` to provide your AWS access credentials
   - Or set environment variables:
     ```
     export AWS_ACCESS_KEY_ID='your_access_key_here'
     export AWS_SECRET_ACCESS_KEY='your_secret_key_here'
     export AWS_REGION='us-east-1'
     ```

3. Make sure you have access to the Mistral AI models in AWS Bedrock in your AWS account.

## Running the Server

Start the server with:
```
python -m modules.server
```

The server will run on port 5001 by default.

## API Endpoints

The API endpoints remain the same as the original implementation:

- `/upload` - Upload a document for processing
- `/job_status/<job_id>` - Check status of a job
- `/job_result/<job_id>` - Get results of a completed job
- `/process_text` - Process text directly (legacy endpoint)

## Architecture

The implementation follows the same architecture as the original, with these key differences:

1. Local model hosting is replaced with AWS Bedrock API calls
2. The prompt format is maintained but sent to Bedrock instead of a local model
3. Rate limiting is implemented to prevent exceeding AWS Bedrock quotas
4. Error handling includes AWS-specific error types

## Monitoring

AWS Bedrock usage can be monitored in the AWS Console. The application also logs information about API calls, including:
- Number of requests
- Processing time
- Success/failure rate 