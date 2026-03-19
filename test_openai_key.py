"""
Simple script to test OpenAI API key validity.

Usage:
    python test_openai_key.py
    python test_openai_key.py --key sk-your-key-here
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_key(api_key: str = None):
    """Test OpenAI API key by making a simple API call."""
    
    # Get API key from argument, environment variable, or .env file
    if api_key:
        key = api_key
    else:
        key = os.getenv('OPENAI_API_KEY')
    
    if not key:
        print("❌ ERROR: No API key found!")
        print("\nOptions:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Add OPENAI_API_KEY to .env file")
        print("  3. Pass key as argument: python test_openai_key.py --key sk-...")
        return False
    
    print("=" * 60)
    print("Testing OpenAI API Key")
    print("=" * 60)
    print(f"Key (first 10 chars): {key[:10]}...{key[-4:]}")
    print()
    
    try:
        from openai import OpenAI
        
        # Initialize client
        client = OpenAI(api_key=key)
        
        print("📡 Making test API call...")
        
        # Make a simple, cheap API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API key is working!' if you can read this."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        
        print("✅ SUCCESS: API key is valid!")
        print(f"   Response: {answer}")
        print(f"   Model: {response.model}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except ImportError:
        print("❌ ERROR: OpenAI package not installed!")
        print("   Install with: pip install openai")
        return False
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"❌ ERROR: {error_type}")
        
        if "401" in error_msg or "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
            print("   API key is INVALID or EXPIRED")
            print("   Get a new key at: https://platform.openai.com/api-keys")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print("   Rate limit exceeded (key is valid but quota exceeded)")
        elif "insufficient_quota" in error_msg.lower():
            print("   Insufficient quota (key is valid but account has no credits)")
        else:
            print(f"   {error_msg}")
        
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenAI API key")
    parser.add_argument('--key', type=str, help='OpenAI API key to test')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Model to test (default: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    success = test_openai_key(args.key)
    
    print()
    print("=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed - check the error message above")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
