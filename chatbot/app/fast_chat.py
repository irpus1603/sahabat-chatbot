"""
Fast Chat Implementation

This module provides a simplified, fast chat implementation that bypasses
complex LangGraph overhead while maintaining database persistence.
"""

import os
import logging
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from .simple_django_checkpointer import get_simple_django_checkpointer

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class FastChatConfig:
    """Fast chat configuration."""
    
    def __init__(self):
        self.endpoint_url = os.getenv('CEDAO_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('CEDAO_LLM_ENDPOINT_TOKEN')
        self.model_name = os.getenv('LLM_MODEL')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKEN', '500'))


class FastAPIClient:
    """Fast API client for LLM requests."""
    
    def __init__(self, config: FastChatConfig):
        self.config = config
    
    def make_request(self, messages: List[Dict]) -> str:
        """Make API request with timeout handling."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            response = requests.post(
                self.config.endpoint_url,
                headers=headers,
                json=payload,
                timeout=30  # More reasonable timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
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
        except requests.exceptions.ConnectionError:
            return "Cannot connect to AI service. Please check connection."
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return f"Error: {str(e)}"


class FastChatManager:
    """
    Fast chat manager with minimal overhead and database persistence.
    """
    
    def __init__(self):
        self.config = FastChatConfig()
        self.api_client = FastAPIClient(self.config)
        
        # Try to get Django checkpointer
        try:
            self.checkpointer = get_simple_django_checkpointer()
            self.has_persistence = True
            logger.info("Initialized fast chat with database persistence")
        except Exception as e:
            logger.warning(f"Database persistence not available: {e}")
            self.checkpointer = None
            self.has_persistence = False
    
    def generate_response(self, prompt: str, chat_id: str, user_context: Optional[Dict] = None) -> str:
        """
        Generate response with minimal processing overhead.
        
        Args:
            prompt: User's input message
            chat_id: Chat session identifier
            user_context: Optional user context
            
        Returns:
            Generated response
        """
        try:
            # Load previous messages from database if available (limit to 4 messages for speed)
            previous_messages = []
            if self.has_persistence and self.checkpointer:
                try:
                    stored_messages = self.checkpointer.load_conversation_state(chat_id)
                    if stored_messages:
                        # Convert to API format - only last 4 messages for context
                        for msg in stored_messages[-4:]:  # Reduced from 6 to 4 for speed
                            if isinstance(msg, HumanMessage):
                                previous_messages.append({"role": "user", "content": msg.content})
                            elif isinstance(msg, AIMessage):
                                previous_messages.append({"role": "assistant", "content": msg.content})
                except Exception as e:
                    logger.warning(f"Failed to load conversation history: {e}")
            
            # Add current message
            messages = previous_messages + [{"role": "user", "content": prompt}]
            
            # Log the message count for debugging
            logger.info(f"Sending {len(messages)} messages to API for chat {chat_id}")
            
            # Generate response
            response = self.api_client.make_request(messages)
            
            # Save to database if available
            if self.has_persistence and self.checkpointer:
                try:
                    # Convert to LangChain format for storage
                    all_messages = []
                    for msg in previous_messages:
                        if msg["role"] == "user":
                            all_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            all_messages.append(AIMessage(content=msg["content"]))
                    
                    # Add current exchange
                    all_messages.append(HumanMessage(content=prompt))
                    all_messages.append(AIMessage(content=response))
                    
                    # Save to database
                    self.checkpointer.save_conversation_state(
                        chat_id, 
                        all_messages, 
                        user_context or {}
                    )
                except Exception as e:
                    logger.warning(f"Failed to save conversation: {e}")
            
            logger.info(f"Generated fast response for chat {chat_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in fast chat generation: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def get_conversation_history(self, chat_id: str, max_messages: int = 20) -> List[Dict]:
        """Get conversation history."""
        if not self.has_persistence or not self.checkpointer:
            return []
        
        try:
            messages = self.checkpointer.load_conversation_state(chat_id)
            if not messages:
                return []
            
            # Convert to API format
            api_messages = []
            for msg in messages[-max_messages:]:
                if isinstance(msg, HumanMessage):
                    api_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    api_messages.append({"role": "assistant", "content": msg.content})
            
            return api_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_conversation(self, chat_id: str) -> bool:
        """Clear conversation history."""
        if not self.has_persistence or not self.checkpointer:
            return True
        
        try:
            return self.checkpointer.clear_conversation_state(chat_id)
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        if self.has_persistence and self.checkpointer:
            stats = self.checkpointer.get_conversation_stats()
            stats.update({
                "memory_type": "Fast Chat + Database",
                "status": "active",
                "persistence": "database-backed"
            })
            return stats
        else:
            return {
                "memory_type": "Fast Chat (No Persistence)",
                "status": "active", 
                "persistence": "none"
            }


class FastSahabatLLM(BaseModel):
    """
    Fast version of SahabatLLM without LangGraph overhead.
    """
    
    endpoint_url: str = Field(..., description="API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key")
    model_name: str = Field(..., description="Model name")
    temperature: float = Field(0.7, description="Temperature")
    max_token: int = Field(200, description="Max tokens")
    chat_manager: Optional[Any] = Field(default=None, description="Chat manager instance")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.chat_manager = FastChatManager()
    
    class Config:
        arbitrary_types_allowed = True
    
    def generate_response(self, prompt: str, chat_id: str, user_context: Optional[Dict] = None) -> str:
        """Generate response using fast chat manager."""
        return self.chat_manager.generate_response(prompt, chat_id, user_context)
    
    def get_conversation_history(self, chat_id: str, max_messages: int = 20) -> List[Dict]:
        """Get conversation history."""
        return self.chat_manager.get_conversation_history(chat_id, max_messages)
    
    def clear_conversation(self, chat_id: str) -> bool:
        """Clear conversation."""
        return self.chat_manager.clear_conversation(chat_id)
    
    def get_memory_stats(self) -> Dict:
        """Get memory stats."""
        return self.chat_manager.get_memory_stats()
    
    def save_conversation(self, chat_id: str, user_message: str, ai_response: str) -> None:
        """Save conversation (compatibility method)."""
        logger.info(f"Conversation automatically saved for chat {chat_id}")


# Global fast chat manager
_global_fast_chat_manager = None


def get_fast_chat_manager() -> FastChatManager:
    """Get global fast chat manager instance."""
    global _global_fast_chat_manager
    if _global_fast_chat_manager is None:
        _global_fast_chat_manager = FastChatManager()
    return _global_fast_chat_manager


def create_fast_llm_instance() -> FastSahabatLLM:
    """Create fast LLM instance."""
    config = FastChatConfig()
    return FastSahabatLLM(
        endpoint_url=config.endpoint_url,
        api_key=config.api_key,
        model_name=config.model_name,
        temperature=config.temperature,
        max_token=config.max_tokens
    )


def fast_health_check() -> Dict:
    """Fast health check."""
    try:
        manager = get_fast_chat_manager()
        test_response = manager.generate_response("Hello", "test_health_check")
        
        return {
            "status": "healthy",
            "memory_type": "Fast Chat",
            "test_response_length": len(test_response),
            "memory_stats": manager.get_memory_stats()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "memory_type": "Fast Chat"
        }