import json
import os
import time
import boto3
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bedrock_integration')

# Get AWS credentials from environment variables
# These should be set in the environment, not hardcoded
os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')

class BedrockClient:
    def __init__(self):
        """Initialize the Bedrock client with credentials"""
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.environ.get('AWS_REGION')
            )
            # Get the model ID from environment variable
            self.model_id = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-8b-instruct-v1:0')
            logger.info(f"Initialized Bedrock client in {os.environ.get('AWS_REGION')}")
            logger.info(f"Using model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def process_chunk(self, chunk, instruction, chunk_id):
        """Process a document chunk using AWS Bedrock Mistral model"""
        try:
            total_chunks = int(chunk_id.split('/')[1]) if '/' in str(chunk_id) else 1
            current_chunk = int(chunk_id.split('/')[0]) if '/' in str(chunk_id) else chunk_id
            
            logger.info(f"Processing chunk {current_chunk}/{total_chunks} with Bedrock")
            
            # Create Mistral-specific prompt format with [INST] and [/INST] tags
            mistral_prompt = f"<s>[INST] "
            # Add system prompt within the instruction
            mistral_prompt += """You are a precise contract editor that modifies legal documents according to user instructions while preserving HTML formatting.

Core requirements:
1. You MUST make ALL changes requested in the user's instructions - this is CRITICAL
2. You MUST maintain the EXACT HTML structure and formatting of the original document
3. ONLY modify the text content within HTML elements as specified in the instructions
4. NEVER add, remove, or modify HTML tags, attributes, or structure
5. Preserve ALL formatting elements: headings, paragraphs, styling, spacing
6. If there are MULTIPLE instructions, implement EACH ONE separately
7. Return the COMPLETE document with HTML formatting intact
8. Make ONLY the text changes specified - do not alter anything else

HTML FORMATTING RULES:
- Keep all <div>, <h1>, <h2>, <p>, and other HTML tags exactly as they are
- Maintain all CSS classes and styling attributes
- Preserve line breaks, spacing, and document structure
- Only change the actual text content between HTML tags
- If the input has HTML structure, your output must have the same HTML structure

IMPORTANT NOTES:
- Pay special attention to company names, addresses, dates, and monetary values
- Look for the specific text mentioned in the instruction and replace it EXACTLY as requested
- Instructions often specify entity names with quotes (e.g., from 'ABC Inc.' to 'XYZ Corp.')
- Maintain exact terminology from the original document for unchanged content
- If you can't find the exact text mentioned, look for similar text that matches the context
"""
            
            mistral_prompt += f"""You are processing chunk {current_chunk} of {total_chunks} of a document.

Original text for chunk {current_chunk}/{total_chunks}: 
"{chunk}"

---
Instruction to apply to this chunk: {instruction}

The instruction may contain MULTIPLE changes to make. Implement ALL of them that apply to this chunk.
Some instructions may not apply to this specific chunk but to other parts of the document.
Return the FULL modified text for this chunk with ALL applicable changes implemented."""
            mistral_prompt += " [/INST]"
            
            # Add debug logging
            logger.info(f"Sending prompt to Bedrock (length: {len(mistral_prompt)})")
            logger.info(f"Chunk preview: {chunk[:200]}...")
            logger.info(f"Instruction: {instruction}")
            
            # Calculate max tokens based on input length - add buffer for the response
            chunk_len = len(chunk)
            max_tokens = min(4000, chunk_len + 500)
            
            # Make the API call to AWS Bedrock
            start_time = time.time()
            
            response = self._call_bedrock_with_retry(
                model_id=self.model_id,  # Use the configured model ID
                prompt=mistral_prompt,
                max_tokens=max_tokens
            )
            
            # Extract the generated text from the response
            # Handle different response formats
            generated_text = ""
            if 'outputs' in response and len(response['outputs']) > 0:
                # Format: {'outputs': [{'text': '...', 'stop_reason': 'stop'}]}
                generated_text = response['outputs'][0].get('text', '')
            elif 'generation' in response:
                # Format: {'generation': '...'}
                generated_text = response.get('generation', '')
            else:
                # Log the actual response format for debugging
                logger.warning(f"Unexpected response format: {response}")
                generated_text = ""
            
            # Log metrics and debugging information
            response_length = len(generated_text) if generated_text else 0
            logger.info(f"Completed chunk {current_chunk}/{total_chunks} - Response length: {response_length} chars in {time.time() - start_time:.2f}s")
            
            # Add detailed response logging
            if generated_text:
                logger.info(f"Response preview: {generated_text[:300]}...")
                
                # Check if the response actually contains changes
                chunk_lower = chunk.lower()
                response_lower = generated_text.lower()
                if chunk_lower != response_lower:
                    logger.info("✅ Response differs from original chunk - changes detected")
                else:
                    logger.warning("⚠️ Response identical to original chunk - no changes detected")
            
            # If response is empty or too short, return the original chunk
            if not generated_text or len(generated_text) < 10:
                logger.warning(f"Empty or very short response for chunk {current_chunk}/{total_chunks}. Using original text.")
                return chunk, False
                
            return generated_text, True
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id} with Bedrock: {str(e)}")
            # Return original chunk on error
            return chunk, False

    def _call_bedrock_with_retry(self, model_id, prompt, max_tokens=4000, temperature=0.4, retries=3):
        """Call AWS Bedrock with retry logic for transient errors"""
        last_exception = None
        backoff_time = 1  # Starting backoff time in seconds
        
        for attempt in range(retries):
            try:
                # Prepare the request body
                request_body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.7,
                    "top_k": 1,
                    "stop": ["</s>"]  # Ministral stop token
                }
                
                # Make the API call
                response = self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                # Parse the response
                response_body = json.loads(response.get('body').read())
                return response_body
                
            except boto3.exceptions.Boto3Error as e:
                last_exception = e
                error_str = str(e)
                
                # Check for credential issues
                if "UnrecognizedClientException" in error_str or "InvalidSignatureException" in error_str or "security token" in error_str.lower():
                    logger.error(f"AWS Credentials Error: {error_str}")
                    logger.error("Please check your AWS credentials and permissions for Bedrock")
                    raise Exception(f"AWS Credentials Error: Invalid or missing AWS credentials. Please configure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and ensure Bedrock permissions.")
                
                # Check for rate limiting or throttling
                elif "ThrottlingException" in error_str or "TooManyRequestsException" in error_str:
                    logger.warning(f"Rate limited by Bedrock (attempt {attempt+1}/{retries}), retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    continue
                    
                # Check for service unavailability
                elif "ServiceUnavailable" in error_str or "InternalServerError" in error_str:
                    logger.warning(f"Bedrock service unavailable (attempt {attempt+1}/{retries}), retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                    
                else:
                    # Other AWS errors, log and retry with backoff
                    logger.error(f"AWS Bedrock error (attempt {attempt+1}/{retries}): {error_str}")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                    
        # If we get here, all retries failed
        logger.error(f"All Bedrock API call attempts failed: {str(last_exception)}")
        raise last_exception 