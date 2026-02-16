import boto3
import os

# Configuration
kb_name = "restaurant-assistant"
AWS_REGION = "us-east-1"

# Get KB ID
ssm_client = boto3.client("ssm", region_name=AWS_REGION)
kb_id_param = ssm_client.get_parameter(Name=f"{kb_name}-kb-id", WithDecryption=False)
kb_id = kb_id_param["Parameter"]["Value"]

print(f"Knowledge Base ID: {kb_id}")
print(f"Region: {AWS_REGION}\n")

# Test KB access
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

try:
    response = bedrock_agent.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={
            "text": "restaurants in San Francisco"
        },
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5
            }
        }
    )
    
    print("✅ Knowledge Base is accessible!")
    print(f"\nFound {len(response.get('retrievalResults', []))} results:")
    
    for i, result in enumerate(response.get('retrievalResults', [])[:3], 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Content: {result.get('content', {}).get('text', 'N/A')[:200]}...")
        
except Exception as e:
    print(f"❌ Error accessing Knowledge Base:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    print("\nPossible issues:")
    print("  1. Knowledge Base is empty (no documents ingested)")
    print("  2. Knowledge Base sync is in progress")
    print("  3. IAM permissions missing for bedrock-agent-runtime")
    print("  4. Knowledge Base ID is incorrect")