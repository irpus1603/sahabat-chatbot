#!/usr/bin/env python3
"""
Test script to debug streaming issues with your LLM API
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot.settings')
django.setup()

from app.chat import create_llm_instance
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_streaming():
    """Test streaming functionality"""
    print("Testing LLM streaming...")
    
    try:
        llm = create_llm_instance()
        
        # Test streaming support
        streaming_supported = llm._api_client.test_streaming_support()
        print(f"Streaming supported: {streaming_supported}")
        
        # Test regular response first
        print("\n=== Testing regular response ===")
        regular_response = llm.generate_response("Hello, how are you?", chat_id="test_123")
        print(f"Regular response: {regular_response[:100]}...")
        
        # Test streaming response
        print("\n=== Testing streaming response ===")
        streaming_response = ""
        chunk_count = 0
        
        for chunk in llm.generate_streaming_response("Hello, how are you?", chat_id="test_456"):
            streaming_response += chunk
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk[:50]}...")
        
        print(f"\nTotal chunks: {chunk_count}")
        print(f"Full streaming response: {streaming_response[:200]}...")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_streaming()