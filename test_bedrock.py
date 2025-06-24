import os
import sys
from nxtAppCore.bedrock_integration.client import get_bedrock_client, invoke_mistral_model, list_available_models

# Set environment variable to use Bedrock
os.environ['USE_BEDROCK'] = 'true'

# Get credentials from environment variables
# Do not hardcode credentials - they should be set in the environment
os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')

def test_bedrock_client():
    """Test that we can initialize the Bedrock client"""
    try:
        client = get_bedrock_client()
        print("✅ Successfully initialized Bedrock client")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Bedrock client: {str(e)}")
        return False

def test_list_models():
    """Test listing available models"""
    try:
        print("\n=== Listing available models ===")
        list_available_models()
        print("✅ Successfully listed models")
        return True
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return False

def test_invoke_model():
    """Test model invocation with both imported and fallback models"""
    success = False
    
    # Try imported model first
    try:
        print("\n=== Testing imported Mistral model ===")
        imported_model_id = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-8x7b-instruct-v0:1')
        prompt = "<s>[INST] Say hello [/INST]"
        result = invoke_mistral_model(prompt, max_tokens=10, modelId=imported_model_id)
        print(f"✅ Successfully invoked imported model. Response: {result}")
        success = True
    except Exception as e:
        print(f"⚠️ Error with imported model: {str(e)}")
        print("Will try fallback model...")
    
    # Try fallback model (Claude)
    try:
        print("\n=== Testing Mistral fallback model ===")
        fallback_model_id = "mistral.mistral-8x7b-instruct-v0:1"
        # Mistral prompt format
        prompt = "<s>[INST] Say hello [/INST]"
        result = invoke_mistral_model(prompt, max_tokens=10, modelId=fallback_model_id)
        print(f"✅ Successfully invoked Mistral fallback model. Response: {result}")
        success = True
    except Exception as e:
        print(f"❌ Error with fallback model: {str(e)}")
    
    return success

if __name__ == "__main__":
    print("Testing Bedrock integration with real credentials...")
    client_test = test_bedrock_client()
    models_test = test_list_models()
    model_test = test_invoke_model()
    
    if client_test and model_test:
        print("\n✅ Bedrock integration is set up correctly and working!")
        print("Test completed successfully - credentials are properly configured")
        sys.exit(0)
    else:
        print("\n❌ Bedrock integration test failed")
        sys.exit(1) 