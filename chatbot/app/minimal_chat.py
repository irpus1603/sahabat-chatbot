"""
Minimal Chat Implementation for Debugging

This bypasses all database operations and complex logic to identify
the root cause of timeouts.
"""

import os
import logging
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MinimalChatManager:
    """Ultra-minimal chat manager for debugging."""
    
    def __init__(self):
        self.endpoint_url = os.getenv('CEDAO_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('CEDAO_LLM_ENDPOINT_TOKEN')
        self.model_name = os.getenv('LLM_MODEL')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKEN', '200'))
    
    def generate_response(self, prompt: str, chat_id: str = None) -> str:
        """Generate response with minimal processing."""
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Prepare payload - ONLY current message, no history
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            logger.info(f"Making API request to {self.endpoint_url}")
            logger.info(f"Payload: {payload}")
            
            # Make request
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Response JSON: {result}")
            
            # Extract response
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    content = choice["message"]["content"]
                    logger.info(f"Extracted content: {content}")
                    return content
                elif "text" in choice:
                    content = choice["text"]
                    logger.info(f"Extracted text: {content}")
                    return content
            elif "response" in result:
                content = result["response"]
                logger.info(f"Extracted response: {content}")
                return content
            
            logger.error(f"Unexpected response format: {result}")
            return "Sorry, I received an unexpected response format."
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return "Request timed out. Please try again."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return "Cannot connect to AI service. Please check connection."
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return f"API error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"


# Global instance
_minimal_chat_manager = None


def get_minimal_chat_manager() -> MinimalChatManager:
    """Get minimal chat manager instance."""
    global _minimal_chat_manager
    if _minimal_chat_manager is None:
        _minimal_chat_manager = MinimalChatManager()
    return _minimal_chat_manager


def test_minimal_chat():
    """Test minimal chat functionality."""
    manager = get_minimal_chat_manager()
    
    print("Testing minimal chat...")
    
    # Test 1
    response1 = manager.generate_response("Hello")
    print(f"Response 1: {response1}")
    
    # Test 2  
    response2 = manager.generate_response("How are you?")
    print(f"Response 2: {response2}")
    
    # Test 3
    response3 = manager.generate_response("What is 2+2?")
    print(f"Response 3: {response3}")


if __name__ == "__main__":
    test_minimal_chat()