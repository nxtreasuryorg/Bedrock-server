import os
import boto3
import json
import time
from botocore.config import Config

# Bedrock client singleton
_bedrock_client = None

def get_bedrock_client():
    """Get or initialize the Bedrock client"""
    global _bedrock_client
    
    if _bedrock_client is None:
        # Get credentials from environment variables
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Check if credentials are available
        if not aws_access_key or not aws_secret_key:
            print("Warning: AWS credentials not found in environment variables.")
            print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            raise EnvironmentError("Missing AWS credentials. Cannot initialize Bedrock client.")
        
        max_attempts = int(os.environ.get('BEDROCK_MODEL_MAX_ATTEMPTS', 10))
        config = Config(
            retries={
                'total_max_attempts': max_attempts,
                'mode': 'standard'
            }
        )
        _bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            config=config
        )
    
    return _bedrock_client

def invoke_mistral_model(prompt, max_tokens=4000, modelId=None):
    """Invoke the model via Bedrock with basic error handling"""
    try:
        client = get_bedrock_client()
        
        # Use provided model ID or default
        if modelId is None:
            modelId = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-8x7b-instruct-v0:1')
        
        print(f"Invoking model: {modelId}")
        
        # Adapt request format based on model type
        if "anthropic" in modelId:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "top_p": 0.7,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            request_body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "top_p": 0.7,
                "top_k": 50,
                "stop": ["</s>", "[/INST]"],
                "return_full_text": False
            }

        response = client.invoke_model(
            modelId=modelId,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get('body').read())
        
        # Extract content based on model type
        if "anthropic" in modelId:
            content = response_body.get('content', [{}])[0].get('text', '')
        else:
            if 'generation' in response_body:
                content = response_body.get('generation', '')
            elif 'outputs' in response_body:
                content = response_body.get('outputs', [{}])[0].get('text', '')
            elif 'text' in response_body:
                content = response_body.get('text', '')
            else:
                print(f"Unexpected response format: {json.dumps(response_body)[:500]}...")
                content = str(response_body)
        
        # Handle empty responses
        if not content or len(content.strip()) < 10:
            print(f"Warning: Model returned empty or very short response")
            content = f"""<!DOCTYPE html>
<html>
<body>
<h1>Model Processing Result</h1>
<p>The model did not generate a useful response for your input.</p>
<p>This could be due to:</p>
<ul>
<li>The model is still initializing</li>
<li>The document format was not properly recognized</li>
<li>The instruction was unclear for the model</li>
</ul>
</body>
</html>"""
        
        return content
        
    except Exception as e:
        print(f"Error calling Bedrock model: {str(e)}")
        # Return a simple error response instead of raising
        return f"""<!DOCTYPE html>
<html>
<body>
<h1>Processing Error</h1>
<p>An error occurred while processing your document: {str(e)}</p>
<p>Please try again or contact support if the issue persists.</p>
</body>
</html>""" 