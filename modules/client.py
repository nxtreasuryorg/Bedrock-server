import os
import boto3
import json
import time
from functools import wraps
from botocore.exceptions import ClientError
from botocore.config import Config

# Bedrock client singleton
_bedrock_client = None

def get_bedrock_client():
    """Get or initialize the Bedrock client"""
    global _bedrock_client
    
    if _bedrock_client is None:
        # Get credentials from environment variables
        # Note: These should be set in the environment or .env file, not hardcoded
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

def with_bedrock_error_handling(func):
    """Decorator to handle Bedrock API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except boto3.exceptions.Boto3Error as e:
                error_type = type(e).__name__
                error_message = str(e)
                
                # Handle specific model not ready error
                if "ModelNotReadyException" in error_message:
                    print(f"Model not ready yet. Using fallback model instead.")
                    # Try with a fallback model if imported model is not ready
                    if "modelId" in kwargs and "imported-model" in kwargs["modelId"]:
                        # Change to Mistral model which is generally available
                        kwargs["modelId"] = "mistral.mistral-8x7b-instruct-v0:1"  # Use Mistral as fallback
                        print(f"Retrying with fallback model: {kwargs['modelId']}")
                        # Continue to next iteration with the new model
                        continue
                    else:
                        # If we already tried with a fallback model and still got this error
                        raise Exception(f"Both primary and fallback models are not ready: {error_message}")
                
                # Handle rate limiting with exponential backoff
                if "ThrottlingException" in error_type or "TooManyRequestsException" in error_type:
                    retry_count += 1
                    if retry_count < max_retries:
                        sleep_time = 2 ** retry_count
                        print(f"Rate limited, retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                        continue
                
                # Handle other errors
                if "AccessDeniedException" in error_type:
                    raise Exception(f"AWS Bedrock access denied. Check credentials and permissions: {error_message}")
                elif "ValidationException" in error_type:
                    raise Exception(f"Invalid request to Bedrock API: {error_message}")
                elif "ServiceUnavailableException" in error_type:
                    raise Exception(f"AWS Bedrock service unavailable: {error_message}")
                else:
                    raise Exception(f"AWS Bedrock error: {error_type} - {error_message}")
            
            # For non-boto3 exceptions
            except Exception as e:
                raise Exception(f"Error calling Bedrock: {str(e)}")
    
    return wrapper

@with_bedrock_error_handling
def invoke_mistral_model(prompt, max_tokens=4000, modelId=None):
    """Invoke the model via Bedrock with ModelNotReadyException retry logic"""
    client = get_bedrock_client()
    
    # Use provided model ID or default to imported Mistral
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
    print(f"Request parameters: {json.dumps(request_body)[:500]}...")

    response = client.invoke_model(
        modelId=modelId,
        body=json.dumps(request_body)
    )
    response_body = json.loads(response.get('body').read())
    print(f"Response keys: {list(response_body.keys())}")
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
    if not content or len(content.strip()) < 10:
        print(f"Warning: Model returned empty or very short response")
        print(f"Response body: {json.dumps(response_body)[:500]}...")
        content = f"""<!DOCTYPE html>\n<html>\n<body>\n<h1>Model Processing Result</h1>\n<p>The model did not generate a useful response for your input.</p>\n<p>This could be due to:</p>\n<ul>\n<li>The model is still initializing</li>\n<li>The document format was not properly recognized</li>\n<li>The instruction was unclear for the model</li>\n</ul>\n</body>\n</html>"""
    return content

def list_available_models():
    """List available models in Bedrock"""
    try:
        client = boto3.client(
            service_name='bedrock',  # Note: not bedrock-runtime
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        # Get foundation models
        foundation_models = client.list_foundation_models()
        print("Available foundation models:")
        for model in foundation_models.get('modelSummaries', []):
            print(f"- {model.get('modelId')}")
            
        # Get imported models
        imported_models = client.list_custom_models()
        print("\nAvailable imported models:")
        for model in imported_models.get('modelSummaries', []):
            print(f"- {model.get('modelArn')}")
            
        return foundation_models, imported_models
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return None, None 