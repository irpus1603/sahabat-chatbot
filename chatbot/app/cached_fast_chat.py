"""
Cached Fast Chat Implementation

This implements in-memory caching and other optimizations for better performance.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
logger = logging.getLogger(__name__)


class CachedFastChatManager:
    """
    Fast chat manager with in-memory caching for improved performance.
    """
    
    def __init__(self):
        self.endpoint_url = os.getenv('CEDAO_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('CEDAO_LLM_ENDPOINT_TOKEN')
        self.model_name = os.getenv('LLM_MODEL')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKEN', '300'))  # Reduced for speed
        
        # In-memory caches
        self.context_cache = {}  # {chat_id: messages}
        self.cache_timestamps = {}  # {chat_id: timestamp}
        self.cache_ttl = 300  # 5 minutes
        
        # Database persistence (optional)
        try:
            from .simple_django_checkpointer import get_simple_django_checkpointer
            self.checkpointer = get_simple_django_checkpointer()
            self.has_persistence = True
        except Exception:
            self.checkpointer = None
            self.has_persistence = False
        
        logger.info("Initialized cached fast chat manager")
    
    def _is_cache_valid(self, chat_id: str) -> bool:
        """Check if cache entry is still valid."""
        if chat_id not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[chat_id]
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl
    
    def _get_context_from_cache(self, chat_id: str) -> List[Dict]:
        """Get conversation context from cache."""
        if chat_id in self.context_cache and self._is_cache_valid(chat_id):
            logger.debug(f"Cache hit for chat {chat_id}")
            return self.context_cache[chat_id]
        
        logger.debug(f"Cache miss for chat {chat_id}")
        return None
    
    def _update_cache(self, chat_id: str, messages: List[Dict]) -> None:
        """Update cache with new messages."""
        # Keep only last 6 messages for efficiency
        self.context_cache[chat_id] = messages[-6:]
        self.cache_timestamps[chat_id] = datetime.now()
        logger.debug(f"Updated cache for chat {chat_id}")
    
    def _load_context_from_database(self, chat_id: str) -> List[Dict]:
        """Load context from database (fallback)."""
        if not self.has_persistence:
            return []
        
        try:
            stored_messages = self.checkpointer.load_conversation_state(chat_id)
            if not stored_messages:
                return []
            
            # Convert to API format - only last 4 messages
            api_messages = []
            for msg in stored_messages[-4:]:  # Reduced from 6 to 4
                if isinstance(msg, HumanMessage):
                    api_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    api_messages.append({"role": "assistant", "content": msg.content})
            
            return api_messages
        except Exception as e:
            logger.warning(f"Failed to load from database: {e}")
            return []
    
    def generate_response(self, prompt: str, chat_id: str, fast_mode: bool = False) -> str:
        """
        Generate response with caching and optional fast mode.
        
        Args:
            prompt: User's message
            chat_id: Chat session ID
            fast_mode: If True, skip conversation context for speed
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare API messages
            if fast_mode:
                # Fast mode: no conversation context
                messages = [{"role": "user", "content": prompt}]
                logger.info(f"Fast mode: no context for chat {chat_id}")
            else:
                # Try cache first
                previous_messages = self._get_context_from_cache(chat_id)
                
                if previous_messages is None:
                    # Cache miss: load from database
                    previous_messages = self._load_context_from_database(chat_id)
                    # Update cache for next time
                    if previous_messages:
                        self._update_cache(chat_id, previous_messages)
                
                # Add current message
                messages = previous_messages + [{"role": "user", "content": prompt}]
                logger.info(f"Context mode: {len(messages)} messages for chat {chat_id}")
            
            # Make API call
            response = self._make_api_call(messages)
            
            # Update cache and database (if not fast mode)
            if not fast_mode:
                # Add response to cache
                updated_messages = messages + [{"role": "assistant", "content": response}]
                self._update_cache(chat_id, updated_messages)
                
                # Save to database in background (non-blocking)
                if self.has_persistence:
                    try:
                        self._save_to_database_async(chat_id, updated_messages)
                    except Exception as e:
                        logger.warning(f"Failed to save to database: {e}")
            
            elapsed = time.time() - start_time
            logger.info(f"Generated response in {elapsed:.2f}s for chat {chat_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I'm experiencing technical difficulties: {str(e)}"
    
    def _make_api_call(self, messages: List[Dict]) -> str:
        """Make optimized API call."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Optimized payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=25  # Reasonable timeout
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
            
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"API Error: {str(e)}"
    
    def _save_to_database_async(self, chat_id: str, messages: List[Dict]) -> None:
        """Save to database (simplified for speed)."""
        try:
            # Convert to LangChain format
            lc_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                else:
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Save to database
            self.checkpointer.save_conversation_state(chat_id, lc_messages)
        except Exception as e:
            logger.warning(f"Background save failed: {e}")
    
    def clear_conversation(self, chat_id: str) -> bool:
        """Clear conversation from cache and database."""
        try:
            # Clear cache
            if chat_id in self.context_cache:
                del self.context_cache[chat_id]
            if chat_id in self.cache_timestamps:
                del self.cache_timestamps[chat_id]
            
            # Clear database
            if self.has_persistence:
                return self.checkpointer.clear_conversation_state(chat_id)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False
    
    def get_memory_stats(self) -> Dict:
        """Get memory and cache statistics."""
        try:
            stats = {
                "memory_type": "Cached Fast Chat",
                "status": "active",
                "cache_entries": len(self.context_cache),
                "cache_ttl_seconds": self.cache_ttl,
            }
            
            if self.has_persistence:
                db_stats = self.checkpointer.get_conversation_stats()
                stats.update(db_stats)
                stats["persistence"] = "database + cache"
            else:
                stats["persistence"] = "cache only"
            
            return stats
        except Exception as e:
            return {"error": str(e)}


# Global instance
_cached_chat_manager = None


def get_cached_chat_manager() -> CachedFastChatManager:
    """Get cached chat manager instance."""
    global _cached_chat_manager
    if _cached_chat_manager is None:
        _cached_chat_manager = CachedFastChatManager()
    return _cached_chat_manager


def test_cached_performance():
    """Test cached chat performance."""
    import time
    
    manager = get_cached_chat_manager()
    chat_id = "cache_test_123"
    
    print("=== Cached Chat Performance Test ===")
    
    # Test 1: First request (cache miss)
    print("\\n1. First request (cache miss)...")
    start_time = time.time()
    response1 = manager.generate_response("Hello, my name is Alice", chat_id)
    time1 = time.time() - start_time
    print(f"First request: {time1:.2f}s - {response1[:50]}...")
    
    # Test 2: Second request (cache hit)
    print("\\n2. Second request (cache hit)...")
    start_time = time.time()
    response2 = manager.generate_response("What is my name?", chat_id)
    time2 = time.time() - start_time
    print(f"Second request: {time2:.2f}s - {response2[:50]}...")
    
    # Test 3: Fast mode (no context)
    print("\\n3. Fast mode request (no context)...")
    start_time = time.time()
    response3 = manager.generate_response("What is 2+2?", chat_id, fast_mode=True)
    time3 = time.time() - start_time
    print(f"Fast mode: {time3:.2f}s - {response3[:50]}...")
    
    # Stats
    stats = manager.get_memory_stats()
    print(f"\\nCache stats: {stats}")
    
    print(f"\\nPerformance comparison:")
    print(f"Cache miss: {time1:.2f}s")
    print(f"Cache hit: {time2:.2f}s")
    print(f"Fast mode: {time3:.2f}s")
    
    if time2 < time1:
        improvement = ((time1 - time2) / time1) * 100
        print(f"Cache improvement: {improvement:.1f}% faster")


if __name__ == "__main__":
    test_cached_performance()