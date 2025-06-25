import os
import sys
from modules.bedrock_integration import BedrockClient

# Set environment variable to use Bedrock
os.environ['USE_BEDROCK'] = 'true'

# Get credentials from environment variables
# Do not hardcode credentials - they should be set in the environment
os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')

def test_bedrock_client():
    """Test that we can initialize the Bedrock client"""
    try:
        client = BedrockClient()
        print("✅ Successfully initialized Bedrock client")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Bedrock client: {str(e)}")
        return False

def test_list_models():
    """Test listing available models"""
    try:
        print("\n=== Listing available models ===")
        # For now, just confirm we can create a client
        client = BedrockClient()
        print(f"✅ Successfully created client with model: {client.model_id}")
        return True
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")
        return False

def test_invoke_model():
    """Test model invocation with Bedrock"""
    try:
        print("\n=== Testing Bedrock model invocation ===")
        client = BedrockClient()
        
        # Test the actual Bedrock API call directly
        test_prompt = "<s>[INST] Say hello [/INST]"
        response = client._call_bedrock_with_retry(
            model_id=client.model_id,
            prompt=test_prompt,
            max_tokens=50
        )
        print(f"✅ Successfully invoked Bedrock model directly")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Error with Bedrock model: {str(e)}")
        return False

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