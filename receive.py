import json
import os
from openai import OpenAI
import dotenv

def read_batch_ids(file_path):
    """Read batch job IDs from the JSONL file."""
    batch_ids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data = json.loads(line)
                    if 'batch_job_id' in data:
                        batch_ids.append(data['batch_job_id'])
        return batch_ids
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {file_path}: {e}")
        return []

def get_batch_status(client, batch_id):
    """Get the status of a batch job."""
    try:
        batch = client.batches.retrieve(batch_id)
        return batch
    except Exception as e:
        print(f"Error retrieving batch {batch_id}: {e}")
        return None

def download_batch_output(client, batch_id, output_dir="batch_outputs"):
    """Download the output file for a completed batch job."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get batch details
        batch = client.batches.retrieve(batch_id)
        
        if batch.status == "completed" and batch.output_file_id:
            # Download the output file
            file_response = client.files.content(batch.output_file_id)
            
            # Save to local file
            output_file_path = os.path.join(output_dir, f"{batch_id}_output.jsonl")
            with open(output_file_path, 'wb') as f:
                f.write(file_response.content)
            
            print(f"✅ Downloaded output for batch {batch_id} to {output_file_path}")
            return output_file_path
        else:
            print(f"⏳ Batch {batch_id} status: {batch.status}")
            if batch.status == "failed" and batch.error_file_id:
                print(f"❌ Batch failed. Downloading error details...")
                download_and_show_errors(client, batch_id, batch.error_file_id, output_dir)
            elif batch.status in ["validating", "in_progress", "finalizing"]:
                print(f"🔄 Batch is still processing...")
            elif batch.status == "expired":
                print(f"⏰ Batch has expired")
            elif batch.status == "cancelled":
                print(f"🚫 Batch was cancelled")
            return None
            
    except Exception as e:
        print(f"Error downloading output for batch {batch_id}: {e}")
        return None

def process_batch_responses(output_file_path):
    """Process the downloaded batch responses."""
    try:
        responses = []
        with open(output_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    response = json.loads(line)
                    responses.append(response)
        
        print(f"📊 Processed {len(responses)} responses from {output_file_path}")
        return responses
    except Exception as e:
        print(f"Error processing responses from {output_file_path}: {e}")
        return []

def download_and_show_errors(client, batch_id, error_file_id, output_dir="batch_outputs"):
    """Download and display error details for a failed batch job."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the error file
        error_response = client.files.content(error_file_id)
        
        # Save to local file
        error_file_path = os.path.join(output_dir, f"{batch_id}_errors.jsonl")
        with open(error_file_path, 'wb') as f:
            f.write(error_response.content)
        
        print(f"📥 Downloaded error file for batch {batch_id} to {error_file_path}")
        
        # Read and display the errors
        print("🔍 Error details:")
        with open(error_file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    error_data = json.loads(line)
                    print(f"  Error {i}:")
                    if 'error' in error_data:
                        error_info = error_data['error']
                        print(f"    Type: {error_info.get('type', 'Unknown')}")
                        print(f"    Code: {error_info.get('code', 'Unknown')}")
                        print(f"    Message: {error_info.get('message', 'No message')}")
                    if 'custom_id' in error_data:
                        print(f"    Custom ID: {error_data['custom_id']}")
                    print()
        
        return error_file_path
    except Exception as e:
        print(f"Error downloading/processing error file for batch {batch_id}: {e}")
        return None

def main():
    """Main function to retrieve and process batch API responses."""
    # Initialize OpenAI client (make sure to set OPENAI_API_KEY environment variable)
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Read batch IDs from file
    batch_ids_file = "batch_job_ids_gpt-4o-mini.jsonl"
    batch_ids = read_batch_ids(batch_ids_file)
    
    if not batch_ids:
        print("No batch IDs found in the file.")
        return
    
    print(f"Found {len(batch_ids)} batch ID(s): {batch_ids}")
    
    # Process each batch
    all_responses = []
    for batch_id in batch_ids:
        print(f"\n🔍 Processing batch: {batch_id}")
        
        # Get batch status
        batch_info = get_batch_status(client, batch_id)
        if batch_info:
            print(f"Status: {batch_info.status}")
            print(f"Created at: {batch_info.created_at}")
            if hasattr(batch_info, 'request_counts'):
                print(f"Request counts: {batch_info.request_counts}")
            
            # Debug: Print all batch_info attributes to see what's available
            print(f"Debug - Full batch info: {batch_info}")
        
        # Download output if completed
        output_file = download_batch_output(client, batch_id)
        if output_file:
            responses = process_batch_responses(output_file)
            all_responses.extend(responses)
        
        # Always check for errors regardless of status (completed batches can still have errors)
        if batch_info:
            if hasattr(batch_info, 'errors') and batch_info.errors and batch_info.errors.data:
                print("🔍 Error details from batch:")
                for i, error in enumerate(batch_info.errors.data, 1):
                    print(f"  Error {i}:")
                    print(f"    Code: {error.code}")
                    print(f"    Line: {error.line}")
                    print(f"    Message: {error.message}")
                    if hasattr(error, 'param') and error.param:
                        print(f"    Param: {error.param}")
                    print()
            
            if hasattr(batch_info, 'error_file_id') and batch_info.error_file_id:
                print("📥 Downloading error file...")
                download_and_show_errors(client, batch_id, batch_info.error_file_id)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"Total batches processed: {len(batch_ids)}")
    print(f"Total responses retrieved: {len(all_responses)}")
    
    # Optional: Save all responses to a single file
    if all_responses:
        combined_output = "all_batch_responses.jsonl"
        with open(combined_output, 'w') as f:
            for response in all_responses:
                f.write(json.dumps(response) + '\n')
        print(f"💾 All responses saved to {combined_output}")

if __name__ == "__main__":
    main()
