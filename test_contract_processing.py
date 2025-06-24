import os
import sys
import base64
from io import BytesIO
from nxtAppCore.bedrock_integration import process_with_bedrock

# Set environment variable to use Bedrock
os.environ['USE_BEDROCK'] = 'true'

# Get credentials from environment variables
# Do not hardcode credentials - they should be set in the environment
os.environ['AWS_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')

# Create a simple test contract file
def create_test_contract():
    """Create a simple test contract file"""
    contract_content = """
    SIMPLE CONSULTING AGREEMENT
    
    This CONSULTING AGREEMENT (the "Agreement") is made and entered into as of January 1, 2024 (the "Effective Date"), 
    by and between ABC Corp., a Delaware corporation with its principal place of business at 123 Main Street, 
    San Francisco, CA 94105 ("Company"), and John Smith, an individual with an address at 456 Oak Avenue, 
    San Jose, CA 95110 ("Consultant").
    
    1. SERVICES
    
    1.1 Services. Consultant shall provide the services described in Exhibit A (the "Services") to Company.
    
    1.2 Performance. Consultant shall perform the Services in a professional manner and in accordance with industry standards.
    
    2. COMPENSATION
    
    2.1 Fees. Company shall pay Consultant the fees set forth in Exhibit A (the "Fees").
    
    2.2 Payment Terms. Company shall pay Consultant within thirty (30) days of receipt of Consultant's invoice.
    
    3. TERM AND TERMINATION
    
    3.1 Term. This Agreement shall commence on the Effective Date and continue for a period of one (1) year, 
    unless earlier terminated as provided herein.
    
    3.2 Termination. Either party may terminate this Agreement upon thirty (30) days' written notice to the other party.
    
    4. CONFIDENTIALITY
    
    4.1 Confidential Information. Consultant shall maintain the confidentiality of all confidential information disclosed 
    by Company.
    
    5. MISCELLANEOUS
    
    5.1 Independent Contractor. Consultant is an independent contractor and not an employee of Company.
    
    5.2 Governing Law. This Agreement shall be governed by the laws of the State of California.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
    
    ABC Corp.                                    John Smith
    
    By: _________________________               By: _________________________
    
    Name: Jane Doe                              Name: John Smith
    
    Title: CEO                                  Date: January 1, 2024
    
    Date: January 1, 2024
    """
    
    test_file_path = "test_contract.txt"
    with open(test_file_path, "w") as f:
        f.write(contract_content)
    
    print(f"Created test contract file: {test_file_path}")
    return test_file_path

def test_contract_processing():
    """Test the contract processing functionality"""
    try:
        # Create test contract
        test_file_path = create_test_contract()
        
        # Define a simple modification instruction
        instruction = "Change the company name from 'ABC Corp.' to 'XYZ Technologies, Inc.' and change the effective date from 'January 1, 2024' to 'March 15, 2024'."
        
        print(f"\n=== Processing test contract with instruction: ===\n{instruction}\n")
        
        # Process the document
        result = process_with_bedrock(instruction, test_file_path, "test_contract.txt")
        
        if 'error' in result and result['error']:
            print(f"\n❌ Error processing contract: {result['message']}")
            return False
        
        print("\n✅ Contract processed successfully!")
        print(f"PDF saved to: {result['pdf_path']}")
        
        # Clean up
        try:
            os.remove(test_file_path)
            print(f"Removed test file: {test_file_path}")
        except:
            pass
        
        return True
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing contract processing with Bedrock integration...")
    success = test_contract_processing()
    
    if success:
        print("\n✅ Contract processing test passed!")
        print("Test completed successfully - credentials are properly configured")
        sys.exit(0)
    else:
        print("\n❌ Contract processing test failed")
        sys.exit(1) 