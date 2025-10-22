"""
Optimized Chat Implementation

This focuses on reducing API response time rather than database optimization.
"""

import os
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class OptimizedChatManager:
    """
    Optimized chat manager focusing on API performance improvements.
    """
    
    def __init__(self):
        self.endpoint_url = os.getenv('CEDAO_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('CEDAO_LLM_ENDPOINT_TOKEN')
        self.model_name = os.getenv('LLM_MODEL')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKEN', '200'))  # Reduced for speed
        
        # Connection pool for reuse
        self.session = None
    
    def generate_response_optimized(self, prompt: str, chat_id: str = None) -> str:
        """
        Generate response with multiple optimizations:
        1. Reduced token count
        2. Simplified prompt
        3. No conversation context for speed
        4. Connection reuse
        """
        try:
            import requests
            
            # Reuse session for connection pooling
            if not hasattr(self, '_session') or self._session is None:
                self._session = requests.Session()
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Optimized payload - minimal tokens, no context
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt[:500]}],  # Truncate input
                "temperature": 0.3,  # Lower temperature for faster generation
                "max_tokens": 100,   # Much lower token limit
                "top_p": 0.8,        # More focused responses
            }
            
            logger.info(f"Making optimized API request (max_tokens=100)")
            
            response = self._session.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=15  # Shorter timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
            elif "response" in result:
                return result["response"]
            
            return "Sorry, I received an unexpected response format."
            
        except Exception as e:
            logger.error(f"Optimized API request failed: {e}")
            return f"Error: {str(e)}"
    
    def generate_response_parallel(self, prompts: List[str]) -> List[str]:
        """
        Generate multiple responses in parallel for better throughput.
        """
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.generate_response_optimized, prompt) for prompt in prompts]
            return [future.result() for future in futures]


def create_optimized_chat_manager() -> OptimizedChatManager:
    """Create optimized chat manager."""
    return OptimizedChatManager()


def test_optimization_comparison():
    """Test different optimization approaches."""
    manager = OptimizedChatManager()
    
    import time
    
    print("=== API Optimization Test ===")
    
    # Test 1: Current implementation
    print("\\n1. Testing current API speed...")
    from app.minimal_chat import get_minimal_chat_manager
    minimal = get_minimal_chat_manager()
    
    start_time = time.time()
    try:
        response1 = minimal.generate_response("Hello")
        time1 = time.time() - start_time
        print(f"Current implementation: {time1:.2f}s - {response1[:50]}...")
    except Exception as e:
        print(f"Current implementation failed: {e}")
        time1 = 30.0
    
    # Test 2: Optimized implementation  
    print("\\n2. Testing optimized API speed...")
    start_time = time.time()
    try:
        response2 = manager.generate_response_optimized("Hello")
        time2 = time.time() - start_time
        print(f"Optimized implementation: {time2:.2f}s - {response2[:50]}...")
    except Exception as e:
        print(f"Optimized implementation failed: {e}")
        time2 = 30.0
    
    # Compare
    if time1 > 0 and time2 > 0:
        improvement = ((time1 - time2) / time1) * 100
        print(f"\\nSpeed improvement: {improvement:.1f}% faster")
        print(f"Time saved: {time1 - time2:.2f} seconds")


if __name__ == "__main__":
    test_optimization_comparison()