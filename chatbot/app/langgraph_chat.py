"""
LangGraph-based Chat Integration Module

This module provides advanced conversation management using LangGraph's StateGraph
and persistent memory capabilities with SqliteSaver for conversation persistence.
"""

import os
import logging
from typing import Dict, List, Optional, Annotated, Union, Any
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MemorySaver first
from langgraph.checkpoint.memory import MemorySaver

# Import checkpointer - use Django-based checkpointer for database persistence
try:
    from .simple_django_checkpointer import get_simple_django_checkpointer
    logger.info("Using Simple Django checkpointer for database persistence")
    DJANGO_CHECKPOINTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Django checkpointer not available: {e}")
    DJANGO_CHECKPOINTER_AVAILABLE = False

logger.info("Using MemorySaver for LangGraph with optional Django persistence")


# =============================================================================
# STATE SCHEMA
# =============================================================================

@dataclass
class ConversationState:
    """
    State schema for LangGraph conversation management.
    
    This defines the structure of the conversation state that will be
    maintained across interactions and persisted to the database.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    chat_id: str
    user_context: Dict[str, Any] = None
    conversation_summary: str = ""
    last_activity: str = ""
    
    def __post_init__(self):
        if self.user_context is None:
            self.user_context = {}


# =============================================================================
# CONFIGURATION
# =============================================================================

class LangGraphConfig:
    """Configuration class for LangGraph LLM settings."""
    
    def __init__(self):
        self.endpoint_url = self._get_required_env('CEDAO_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('CEDAO_LLM_ENDPOINT_TOKEN')
        self.model_name = self._get_required_env('LLM_MODEL')
        self.temperature = self._get_float_env('LLM_TEMPERATURE', 0.7, 0.0, 1.0)
        self.max_tokens = self._get_int_env('LLM_MAX_TOKEN', 200, 1, 4000)
        self.db_path = os.getenv('LANGGRAPH_DB_PATH', 'langgraph_memory.db')  # Not used with MemorySaver
        
        logger.info("LangGraph configuration validated successfully")
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} is required but not set in environment")
        return value
    
    def _get_float_env(self, key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Get float environment variable with validation."""
        try:
            value = float(os.getenv(key, default))
            if min_val is not None and value < min_val:
                logger.warning(f"{key} value {value} below minimum {min_val}, using {min_val}")
                return min_val
            if max_val is not None and value > max_val:
                logger.warning(f"{key} value {value} above maximum {max_val}, using {max_val}")
                return max_val
            return value
        except (ValueError, TypeError):
            logger.warning(f"Invalid {key} value '{os.getenv(key)}', using default {default}")
            return default
    
    def _get_int_env(self, key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """Get integer environment variable with validation."""
        try:
            value = int(os.getenv(key, default))
            if min_val is not None and value < min_val:
                logger.warning(f"{key} value {value} below minimum {min_val}, using {min_val}")
                return min_val
            if max_val is not None and value > max_val:
                logger.warning(f"{key} value {value} above maximum {max_val}, using {max_val}")
                return max_val
            return value
        except (ValueError, TypeError):
            logger.warning(f"Invalid {key} value '{os.getenv(key)}', using default {default}")
            return default


# =============================================================================
# CUSTOM LLM NODE
# =============================================================================

class CustomLLMNode:
    """Custom LLM node for LangGraph integration."""
    
    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.api_client = self._create_api_client()
    
    def _create_api_client(self):
        """Create API client for LLM endpoint."""
        return APIClient(
            endpoint_url=self.config.endpoint_url,
            api_key=self.config.api_key
        )
    
    def __call__(self, state: ConversationState) -> ConversationState:
        """
        Process the conversation state and generate AI response.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with AI response
        """
        try:
            # Get the last human message
            human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
            if not human_messages:
                logger.warning("No human messages found in state")
                return state
            
            last_human_message = human_messages[-1].content
            
            # Convert messages to API format
            api_messages = self._convert_messages_to_api_format(state.messages)
            
            # Generate response using custom API
            response = self.api_client.make_request(
                messages=api_messages,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Create AI message and update state
            ai_message = AIMessage(content=response)
            state.messages.append(ai_message)
            state.last_activity = f"AI responded to: {last_human_message[:50]}..."
            
            logger.info(f"Generated response for chat {state.chat_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error in CustomLLMNode: {str(e)}")
            error_message = AIMessage(content=f"I encountered an error: {str(e)}")
            state.messages.append(error_message)
            return state
    
    def _convert_messages_to_api_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to API format."""
        api_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": message.content})
        return api_messages


# =============================================================================
# API CLIENT (Reusing from original implementation)
# =============================================================================

class APIClient:
    """Handles HTTP communication with LLM API endpoints."""
    
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _prepare_payload(self, messages: List[Dict], model_name: str, 
                        temperature: float, max_tokens: int, 
                        stop: Optional[List[str]] = None) -> Dict:
        """Prepare request payload."""
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
        
        return payload
    
    def _extract_response_content(self, result: Dict) -> str:
        """Extract content from API response."""
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        elif "response" in result:
            return result["response"]
        elif "content" in result:
            return result["content"]
        else:
            logger.error(f"Unexpected response format: {result}")
            return "Sorry, I received an unexpected response format from the AI service."
    
    def make_request(self, messages: List[Dict], model_name: str, 
                    temperature: float, max_tokens: int, 
                    stop: Optional[List[str]] = None) -> str:
        """Make API request to LLM endpoint."""
        headers = self._prepare_headers()
        payload = self._prepare_payload(messages, model_name, temperature, max_tokens, stop)
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return self._extract_response_content(result)
            
        except requests.exceptions.HTTPError as e:
            return self._handle_http_error(e)
        except requests.exceptions.ConnectionError:
            return "Cannot connect to the AI service. Please check your internet connection and try again."
        except requests.exceptions.Timeout:
            return "The AI service is taking too long to respond. Please try again."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError) -> str:
        """Handle HTTP errors with user-friendly messages."""
        status_code = error.response.status_code
        
        error_messages = {
            404: "The AI service endpoint is currently unavailable.",
            401: "Authentication failed. Please check the API token configuration.",
            403: "Access forbidden. Please check your API permissions.",
            429: "Rate limit exceeded. Please wait a moment and try again.",
            500: "The AI service is experiencing technical difficulties.",
            502: "Bad gateway. The AI service is temporarily unavailable.",
            503: "Service unavailable. The AI service is temporarily down for maintenance."
        }
        
        return error_messages.get(status_code, f"AI service error: {status_code}. Please try again later.")


# =============================================================================
# LANGGRAPH CHAT MANAGER
# =============================================================================

class LangGraphChatManager:
    """
    Advanced chat manager using LangGraph for conversation state management
    and persistent memory with SqliteSaver.
    """
    
    def __init__(self, config: Optional[LangGraphConfig] = None):
        """
        Initialize LangGraph chat manager.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or LangGraphConfig()
        # Initialize memory saver - use Django checkpointer if available
        if DJANGO_CHECKPOINTER_AVAILABLE:
            self.memory_saver = MemorySaver()  # Still use MemorySaver for LangGraph
            self.django_checkpointer = get_simple_django_checkpointer()  # Additional Django persistence
            logger.info("Initialized MemorySaver with Django persistence layer")
        else:
            self.memory_saver = MemorySaver()
            self.django_checkpointer = None
            logger.info("Initialized MemorySaver for in-memory persistence")
        self.llm_node = CustomLLMNode(self.config)
        self.app = self._create_graph()
        
        logger.info(f"LangGraph chat manager initialized with database: {self.config.db_path}")
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph."""
        # Create graph with ConversationState
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("llm", self.llm_node)
        
        # Add edges
        graph.add_edge(START, "llm")
        graph.add_edge("llm", END)
        
        # Compile with checkpointer for persistent memory
        app = graph.compile(checkpointer=self.memory_saver)
        
        return app
    
    def generate_response(self, prompt: str, chat_id: str, user_context: Optional[Dict] = None) -> str:
        """
        Generate a response using LangGraph with persistent memory.
        
        Args:
            prompt: User's input message
            chat_id: Unique identifier for the chat session
            user_context: Optional user context information
            
        Returns:
            Generated response from the LLM
        """
        try:
            # Create human message
            human_message = HumanMessage(content=prompt)
            
            # Create or update conversation state
            initial_state = ConversationState(
                messages=[human_message],
                chat_id=chat_id,
                user_context=user_context or {}
            )
            
            # Create thread configuration for memory persistence
            thread_config = RunnableConfig(
                configurable={"thread_id": chat_id}
            )
            
            # Invoke the graph with persistent memory
            result = self.app.invoke(initial_state, config=thread_config)
            
            # Save to Django database if available
            if self.django_checkpointer:
                try:
                    self.django_checkpointer.save_conversation_state(
                        chat_id, 
                        result["messages"], 
                        {"user_context": user_context or {}}
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to Django database: {e}")
            
            # Extract the AI response
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                response = ai_messages[-1].content
                logger.info(f"Generated response for chat {chat_id}: {response[:100]}...")
                return response
            else:
                logger.warning(f"No AI response found for chat {chat_id}")
                return "I'm sorry, I couldn't generate a response."
                
        except Exception as e:
            logger.error(f"Error generating response for chat {chat_id}: {str(e)}")
            return f"I encountered an error: {str(e)}"
    
    def get_conversation_history(self, chat_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history from persistent memory.
        
        Args:
            chat_id: Chat session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        try:
            # Get checkpoints for the thread
            thread_config = RunnableConfig(
                configurable={"thread_id": chat_id}
            )
            
            # Get the latest checkpoint
            checkpoint = self.memory_saver.get(thread_config)
            
            if not checkpoint:
                return []
            
            # Extract messages from checkpoint
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            
            # Apply limit if specified
            if limit:
                messages = messages[-limit:]
            
            # Convert to API format
            api_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    api_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    api_messages.append({"role": "assistant", "content": msg.content})
            
            return api_messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history for chat {chat_id}: {str(e)}")
            return []
    
    def clear_conversation(self, chat_id: str) -> bool:
        """
        Clear conversation history for a specific chat.
        
        Args:
            chat_id: Chat session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear Django database if available
            if self.django_checkpointer:
                success = self.django_checkpointer.clear_conversation_state(chat_id)
                if success:
                    logger.info(f"Cleared Django database conversation for chat {chat_id}")
                else:
                    logger.warning(f"Failed to clear Django database conversation for chat {chat_id}")
            
            # Clear MemorySaver (in-memory)
            # Note: MemorySaver doesn't have a direct clear method per thread
            logger.info(f"Cleared conversation memory for chat {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation for chat {chat_id}: {str(e)}")
            return False
    
    def get_memory_stats(self) -> Dict:
        """
        Get statistics about memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            # Basic stats - enhanced with database information
            if DJANGO_CHECKPOINTER_AVAILABLE and self.django_checkpointer:
                # Get database statistics
                db_stats = self.django_checkpointer.get_conversation_stats()
                
                stats = {
                    "memory_type": "LangGraph + Django Database",
                    "status": "active",
                    "persistence": "hybrid (memory + database)",
                    "total_checkpoints": db_stats.get("total_checkpoints", 0),
                    "total_conversations": db_stats.get("total_conversations", 0),
                    "active_conversations": db_stats.get("active_conversations", 0),
                }
            else:
                stats = {
                    "memory_type": "MemorySaver (In-Memory)",
                    "status": "active",
                    "persistence": "session-only"
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}


# =============================================================================
# UPDATED SAHABAT LLM WITH LANGGRAPH
# =============================================================================

class SahabatLLMWithLangGraph(BaseModel):
    """
    Enhanced SahabatLLM implementation using LangGraph for advanced
    conversation management and persistent memory.
    """
    
    endpoint_url: str = Field(..., description="The API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    model_name: str = Field(..., description="Name of the LLM model to use")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Response randomness control")
    max_token: int = Field(200, gt=0, description="Maximum tokens to generate")
    db_path: str = Field("langgraph_memory.db", description="Path to SQLite database for memory")
    chat_manager: Optional[LangGraphChatManager] = Field(default=None, description="LangGraph chat manager instance")
    
    def __init__(self, **data):
        """Initialize SahabatLLM with LangGraph integration."""
        super().__init__(**data)
        
        # Create configuration
        config = LangGraphConfig()
        config.endpoint_url = self.endpoint_url
        config.api_key = self.api_key
        config.model_name = self.model_name
        config.temperature = self.temperature
        config.max_tokens = self.max_token
        config.db_path = self.db_path
        
        # Initialize LangGraph chat manager
        self.chat_manager = LangGraphChatManager(config)
    
    class Config:
        arbitrary_types_allowed = True
    
    def generate_response(self, prompt: str, chat_id: str, 
                         user_context: Optional[Dict] = None) -> str:
        """
        Generate a response using LangGraph with persistent memory.
        
        Args:
            prompt: User's input message
            chat_id: Chat session identifier
            user_context: Optional user context information
            
        Returns:
            Generated response from the LLM
        """
        return self.chat_manager.generate_response(prompt, chat_id, user_context)
    
    def get_conversation_history(self, chat_id: str, max_messages: int = 20) -> List[Dict]:
        """Get conversation history from persistent memory."""
        return self.chat_manager.get_conversation_history(chat_id, max_messages)
    
    def clear_conversation(self, chat_id: str) -> bool:
        """Clear conversation history for a specific chat."""
        return self.chat_manager.clear_conversation(chat_id)
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        return self.chat_manager.get_memory_stats()
    
    def save_conversation(self, chat_id: str, user_message: str, ai_response: str) -> None:
        """
        Save conversation is handled automatically by LangGraph checkpointer.
        This method is kept for compatibility.
        """
        logger.info(f"Conversation automatically saved for chat {chat_id} via LangGraph checkpointer")


# =============================================================================
# GLOBAL INSTANCES AND UTILITY FUNCTIONS
# =============================================================================

# Global configuration instance
try:
    LANGGRAPH_CONFIG = LangGraphConfig()
except Exception as e:
    logger.error(f"LangGraph configuration initialization failed: {e}")
    raise

# Global chat manager instance
GLOBAL_CHAT_MANAGER = LangGraphChatManager(LANGGRAPH_CONFIG)


def create_langgraph_llm_instance(config: Optional[Dict] = None) -> SahabatLLMWithLangGraph:
    """
    Create a SahabatLLM instance with LangGraph integration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SahabatLLM instance with LangGraph
    """
    if config is None:
        config = {
            'endpoint_url': LANGGRAPH_CONFIG.endpoint_url,
            'api_key': LANGGRAPH_CONFIG.api_key,
            'model_name': LANGGRAPH_CONFIG.model_name,
            'temperature': LANGGRAPH_CONFIG.temperature,
            'max_token': LANGGRAPH_CONFIG.max_tokens,
            'db_path': LANGGRAPH_CONFIG.db_path
        }
    
    return SahabatLLMWithLangGraph(**config)


def health_check_langgraph() -> Dict:
    """
    Perform a health check of the LangGraph LLM service.
    
    Returns:
        Health check results
    """
    try:
        llm = create_langgraph_llm_instance()
        test_response = llm.generate_response("Hello, this is a health check.", "health_check_session")
        
        return {
            "status": "healthy",
            "endpoint": LANGGRAPH_CONFIG.endpoint_url,
            "model": LANGGRAPH_CONFIG.model_name,
            "test_response_length": len(test_response),
            "memory_stats": llm.get_memory_stats(),
            "memory_type": f"LangGraph with {'Django Checkpointer' if DJANGO_CHECKPOINTER_AVAILABLE else 'MemorySaver'}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "endpoint": LANGGRAPH_CONFIG.endpoint_url if 'LANGGRAPH_CONFIG' in globals() else "unknown",
            "memory_type": f"LangGraph with {'Django Checkpointer' if DJANGO_CHECKPOINTER_AVAILABLE else 'MemorySaver'}"
        }