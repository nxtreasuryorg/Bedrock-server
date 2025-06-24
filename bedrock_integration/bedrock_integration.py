import json
import os
import time
import boto3
import logging

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
            logger.info(f"Initialized Bedrock client in {os.environ.get('AWS_REGION')}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    def process_chunk(self, chunk, instruction, chunk_id):
        """Process a document chunk using AWS Bedrock Ministral 8B model"""
        try:
            total_chunks = int(chunk_id.split('/')[1]) if '/' in str(chunk_id) else 1
            current_chunk = int(chunk_id.split('/')[0]) if '/' in str(chunk_id) else chunk_id
            
            logger.info(f"Processing chunk {current_chunk}/{total_chunks} with Bedrock")
            
            # Create Mistral-specific prompt format with [INST] and [/INST] tags
            mistral_prompt = f"<s>[INST] "
            # Add system prompt within the instruction
            mistral_prompt += """You are a precise contract editor that strictly modifies legal documents according to user instructions.

Core requirements:
1. You MUST make ALL changes requested in the user's instructions - this is CRITICAL
2. If there are MULTIPLE instructions (separated by line breaks or numbers), you MUST implement EACH ONE separately
3. Make ONLY the changes specified in the instructions - do not add, remove, or modify anything else
4. NEVER generate or invent new legal language not present in the original document
5. ONLY modify text explicitly mentioned in the user's instructions
6. Maintain exact terminology from the original document - do not substitute or paraphrase legal terms
7. Process each instruction step by step - do not skip any requested change
8. Return the FULL modified text, with ALL the requested changes implemented

IMPORTANT NOTES:
- Pay special attention to company names, addresses, dates, and monetary values
- Look carefully for the specific text mentioned in the instruction and replace it EXACTLY as requested
- If an instruction says to change text from 'X' to 'Y', you MUST find and replace every instance of 'X' with 'Y'
- Instructions often specify entity names with quotes (e.g., from 'ABC Inc.' to 'XYZ Corp.') - these are critical to replace correctly
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
            
            # Calculate max tokens based on input length - add buffer for the response
            chunk_len = len(chunk)
            max_tokens = min(4000, chunk_len + 500)
            
            # Make the API call to AWS Bedrock
            start_time = time.time()
            
            response = self._call_bedrock_with_retry(
                model_id="mistral.mistral-8b-instruct-v1:0",  # Ministral 8B
                prompt=mistral_prompt,
                max_tokens=max_tokens
            )
            
            # Extract the generated text from the response
            generated_text = response.get('generation', '')
            
            # Log metrics and debugging information
            response_length = len(generated_text) if generated_text else 0
            logger.info(f"Completed chunk {current_chunk}/{total_chunks} - Response length: {response_length} chars in {time.time() - start_time:.2f}s")
            
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
                
                # Check for rate limiting or throttling
                if "ThrottlingException" in str(e) or "TooManyRequestsException" in str(e):
                    logger.warning(f"Rate limited by Bedrock (attempt {attempt+1}/{retries}), retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                    continue
                    
                # Check for service unavailability
                elif "ServiceUnavailable" in str(e) or "InternalServerError" in str(e):
                    logger.warning(f"Bedrock service unavailable (attempt {attempt+1}/{retries}), retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                    
                else:
                    # Other AWS errors, log and retry with backoff
                    logger.error(f"AWS Bedrock error (attempt {attempt+1}/{retries}): {str(e)}")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                    
        # If we get here, all retries failed
        logger.error(f"All Bedrock API call attempts failed: {str(last_exception)}")
        raise last_exception 