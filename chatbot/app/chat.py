"""
LLM Chat Integration Module

This module provides integration with custom LLM APIs and conversation memory management
using LangChain. It handles API communication, conversation context, and memory persistence.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Generator

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports for conversation memory
# In LangChain 1.x, ConversationBufferMemory is in langchain.memory
# but that module doesn't expose it directly. We need to import from the legacy location
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

try:
    # Try newer import path first (for older versions)
    from langchain.memory import ConversationBufferMemory
except ImportError:
    # For LangChain 1.x, create a compatibility wrapper for ConversationBufferMemory
    class ConversationBufferMemory:
        """Compatibility wrapper for ConversationBufferMemory in LangChain 1.x"""
        def __init__(self, return_messages=True, memory_key="chat_history"):
            self.return_messages = return_messages
            self.memory_key = memory_key
            self.chat_memory = ChatMessageHistory()

        def save_context(self, inputs, outputs):
            """Save context from this conversation to buffer."""
            if "input" in inputs:
                self.chat_memory.add_user_message(inputs["input"])
            if "output" in outputs:
                self.chat_memory.add_ai_message(outputs["output"])

        def load_memory_variables(self, inputs):
            """Return history buffer."""
            if self.return_messages:
                return {self.memory_key: self.chat_memory.messages}
            else:
                return {self.memory_key: "\\n".join([m.content for m in self.chat_memory.messages])}


# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration class for LLM settings with validation."""
    
    def __init__(self):
        self.endpoint_url = self._get_required_env('NOKIA_LLM_ENDPOINT_URL')
        self.api_key = os.getenv('NOKIA_LLM_ENDPOINT_TOKEN')
        self.model_name = self._get_required_env('NOKIA_LLM_MODEL')
        self.temperature = self._get_float_env('LLM_TEMPERATURE', 0.7, 0.0, 1.0)
        self.max_tokens = self._get_int_env('LLM_MAX_TOKEN', 200, 1, 4000)
        
        logger.info("LLM configuration validated successfully")
    
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
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'endpoint_url': self.endpoint_url,
            'api_key': self.api_key,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_token': self.max_tokens
        }


# Global configuration instance
try:
    CONFIG = LLMConfig()
    ENV_CONFIG = CONFIG.to_dict()
except Exception as e:
    logger.error(f"Configuration initialization failed: {e}")
    raise


# =============================================================================
# CONVERSATION MEMORY MANAGEMENT
# =============================================================================

class ConversationMemoryManager:
    """
    Manages conversation memory using LangChain for chat sessions.
    
    Provides memory management per chat session with efficient context retrieval
    and automatic cleanup capabilities.
    """
    
    def __init__(self, max_context_messages: int = 10):
        """
        Initialize conversation memory manager.
        
        Args:
            max_context_messages: Maximum number of messages to keep in context
        """
        self.max_context_messages = max_context_messages
        self._memories: Dict[str, ConversationBufferMemory] = {}
    
    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        """Get or create memory for a specific chat session."""
        if chat_id not in self._memories:
            self._memories[chat_id] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        return self._memories[chat_id]
    
    def add_message(self, chat_id: str, human_message: str, ai_message: str) -> None:
        """Add a conversation exchange to memory."""
        memory = self.get_memory(chat_id)
        memory.chat_memory.add_user_message(human_message)
        memory.chat_memory.add_ai_message(ai_message)
    
    def get_conversation_context(self, chat_id: str, max_messages: Optional[int] = None) -> List[Dict]:
        """
        Get conversation context formatted for API requests.

        Args:
            chat_id: Unique identifier for the chat session
            max_messages: Maximum number of recent messages to include

        Returns:
            List of message dictionaries in API format (without system message)
        """
        memory = self.get_memory(chat_id)
        messages = memory.chat_memory.messages

        # Use provided max_messages or instance default
        limit = max_messages or self.max_context_messages
        recent_messages = messages[-limit:] if limit and len(messages) > limit else messages

        # Convert to API format (excluding system message)
        api_messages = []
        for message in recent_messages:
            if isinstance(message, HumanMessage):
                api_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                api_messages.append({"role": "assistant", "content": message.content})

        return api_messages
    
    def clear_memory(self, chat_id: str) -> None:
        """Clear memory for a specific chat session."""
        if chat_id in self._memories:
            del self._memories[chat_id]
            logger.info(f"Cleared memory for chat {chat_id}")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about current memory usage."""
        stats = {
            'total_chats': len(self._memories),
            'chat_message_counts': {}
        }
        
        for chat_id, memory in self._memories.items():
            stats['chat_message_counts'][chat_id] = len(memory.chat_memory.messages)
        
        return stats


# =============================================================================
# LLM API CLIENT
# =============================================================================

class APIClient:
    """Handles HTTP communication with LLM API endpoints."""
    
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize API client.
        
        Args:
            endpoint_url: The API endpoint URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.endpoint_url.startswith("http://10.34.161.90"):
            headers["Authorization"] = f"Bearer {self.api_key}"
            logger.debug(f"Using Bearer token: {self.api_key[:10]}...")
        return headers
    
    def _prepare_payload(self, messages: List[Dict], model_name: str,
                        temperature: float, max_tokens: int,
                        stop: Optional[List[str]] = None, stream: bool = False) -> Dict:
        """Prepare request payload."""
        # Convert message list to a single prompt string
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        return payload
    
    def _extract_response_content(self, result: Dict) -> str:
        """Extract content from API response."""
        content = None

        # Handle different response formats
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]
        elif "response" in result:
            content = result["response"]
        elif "content" in result:
            content = result["content"]
        else:
            logger.error(f"Unexpected response format: {result}")
            return "Sorry, I received an unexpected response format from the AI service."

        # Clean up response content if it contains conversation history
        # Some LLM APIs echo back the input messages with "user", "model" labels
        if content and "\nmodel\n" in content:
            # Extract only the model's response after "model\n"
            parts = content.split("\nmodel\n")
            if len(parts) > 1:
                content = parts[-1].strip()
                logger.debug(f"Cleaned response content by removing conversation history")

        return content
    
    def make_request(self, messages: List[Dict], model_name: str, 
                    temperature: float, max_tokens: int, 
                    stop: Optional[List[str]] = None) -> str:
        """
        Make API request to LLM endpoint.
        
        Args:
            messages: Conversation messages
            model_name: Model to use
            temperature: Response randomness
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            Generated response content
            
        Raises:
            Various exceptions for different error conditions
        """
        headers = self._prepare_headers()
        payload = self._prepare_payload(messages, model_name, temperature, max_tokens, stop)
        
        logger.info(f"Making request to: {self.endpoint_url}")
        logger.debug(f"Context length: {len(messages)} messages")
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            logger.info(f"Request payload prompt: {payload['prompt'][:200]}...")
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()

            logger.info(f"Raw API response: {json.dumps(result, indent=2)}")

            extracted_content = self._extract_response_content(result)
            logger.info(f"Extracted response content: {extracted_content[:200]}...")

            return extracted_content
            
        except requests.exceptions.HTTPError as e:
            return self._handle_http_error(e)
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed to {self.endpoint_url}")
            return "Cannot connect to the AI service. Please check your internet connection and try again."
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout to {self.endpoint_url}")
            return "The AI service is taking too long to respond. Please try again."
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError) -> str:
        """Handle HTTP errors with user-friendly messages."""
        status_code = error.response.status_code
        
        error_messages = {
            404: "The AI service endpoint is currently unavailable. Please check the configuration or try again later.",
            401: "Authentication failed. Please check the API token configuration.",
            403: "Access forbidden. Please check your API permissions.",
            429: "Rate limit exceeded. Please wait a moment and try again.",
            500: "The AI service is experiencing technical difficulties. Please try again later.",
            502: "Bad gateway. The AI service is temporarily unavailable.",
            503: "Service unavailable. The AI service is temporarily down for maintenance."
        }
        
        message = error_messages.get(status_code, f"AI service error: {status_code}. Please try again later.")
        logger.error(f"HTTP {status_code} error: {error}")
        return message
    
    def test_streaming_support(self) -> bool:
        """Test if the API endpoint supports streaming."""
        try:
            test_payload = {
                "model": "test",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
                "max_tokens": 1
            }
            headers = self._prepare_headers()
            
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=test_payload,
                timeout=5,
                stream=True
            )
            
            # Check if we get a streaming response
            content_type = response.headers.get('content-type', '')
            return 'text/event-stream' in content_type or response.status_code != 400
            
        except Exception:
            return False
    
    def make_streaming_request(self, messages: List[Dict], model_name: str, 
                              temperature: float, max_tokens: int, 
                              stop: Optional[List[str]] = None) -> Generator[str, None, None]:
        """
        Make streaming API request to LLM endpoint.
        
        Args:
            messages: Conversation messages
            model_name: Model to use
            temperature: Response randomness
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Yields:
            Generated response content chunks
            
        Raises:
            Various exceptions for different error conditions
        """
        headers = self._prepare_headers()
        payload = self._prepare_payload(messages, model_name, temperature, max_tokens, stop, stream=True)
        
        logger.info(f"Making streaming request to: {self.endpoint_url}")
        logger.debug(f"Context length: {len(messages)} messages")
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            
            response.raise_for_status()
            
            # Check if response is actually streaming
            content_type = response.headers.get('content-type', '')
            logger.debug(f"Response content-type: {content_type}")
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    logger.debug(f"Received line: {line[:100]}...")  # Log first 100 chars

                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            logger.debug("Received [DONE] signal")
                            break
                        if data.strip() == '':
                            continue  # Skip empty data lines
                        try:
                            chunk = json.loads(data)
                            # Handle Ollama format (response field)
                            if 'response' in chunk:
                                content = chunk.get('response', '')
                                if content:
                                    yield content
                            # Handle OpenAI format (choices array)
                            elif 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON: {data[:100]}... Error: {e}")
                            continue
                    else:
                        # Handle non-SSE format - some APIs return plain JSON lines
                        try:
                            chunk = json.loads(line)
                            # Handle Ollama format (response field)
                            if 'response' in chunk:
                                content = chunk.get('response', '')
                                done = chunk.get('done', False)
                                if content:
                                    yield content
                                if done:
                                    logger.debug("Received done=true from Ollama")
                                    break
                            # Handle OpenAI format (choices array)
                            elif 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logger.debug(f"Non-JSON line (possibly connection keep-alive): {line[:50]}...")
                            continue
                            
        except requests.exceptions.HTTPError as e:
            yield self._handle_http_error(e)
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed to {self.endpoint_url}")
            yield "Cannot connect to the AI service. Please check your internet connection and try again."
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout to {self.endpoint_url}")
            yield "The AI service is taking too long to respond. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            yield f"An unexpected error occurred: {str(e)}"


# =============================================================================
# MAIN LLM CLASS
# =============================================================================

class SahabatLLM(BaseModel):
    """
    Custom LLM implementation with conversation memory and robust error handling.
    
    This class provides a clean interface for LLM interactions with built-in
    conversation memory management and comprehensive error handling.
    """
    
    endpoint_url: str = Field(..., description="The API endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    model_name: str = Field(..., description="Name of the LLM model to use")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Response randomness control")
    max_token: int = Field(200, gt=0, description="Maximum tokens to generate")
    memory_manager: Optional[ConversationMemoryManager] = Field(
        default=None, description="Conversation memory manager"
    )
    
    def __init__(self, **data):
        """Initialize SahabatLLM with automatic memory manager setup."""
        super().__init__(**data)
        if self.memory_manager is None:
            self.memory_manager = ConversationMemoryManager()
        
        self._api_client = APIClient(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key
        )
    
    class Config:
        arbitrary_types_allowed = True
    
    def generate_response(self, prompt: str, chat_id: Optional[str] = None, 
                         stop: Optional[List[str]] = None) -> str:
        """
        Generate a response with conversation context.
        
        Args:
            prompt: User's input message
            chat_id: Chat session identifier for memory context
            stop: Stop sequences for generation
            
        Returns:
            Generated response from the LLM
        """
        # Build conversation context
        messages = self._build_message_context(prompt, chat_id)
        #logger.info(f"get messages from db: {messages}")
        # Generate response
        response = self._api_client.make_request(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_token,
            stop=stop
        )
        
        # Save conversation to memory
        if chat_id and not response.startswith(("Error:", "Cannot connect", "The AI service", "Sorry,")):
            self.save_conversation(chat_id, prompt, response)
        
        return response
    
    def generate_streaming_response(self, prompt: str, chat_id: Optional[str] = None,
                                   stop: Optional[List[str]] = None) -> Generator[str, None, None]:
        """
        Generate a streaming response with conversation context.
        Fallback to regular response if streaming fails.

        Args:
            prompt: User's input message
            chat_id: Chat session identifier for memory context
            stop: Stop sequences for generation

        Yields:
            Generated response content chunks
        """
        # Build conversation context
        messages = self._build_message_context(prompt, chat_id)

        try:
            # Try streaming first
            full_response = ""
            chunk_count = 0

            for chunk in self._api_client.make_streaming_request(
                messages=messages,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_token,
                stop=stop
            ):
                # Accumulate full response
                full_response += chunk
                chunk_count += 1

                # Yield each chunk immediately for streaming
                yield chunk

            # If no chunks received, fall back to regular response
            if chunk_count == 0:
                logger.warning("No streaming chunks received, falling back to regular response")
                full_response = self.generate_response(prompt, chat_id, stop)
                yield full_response

            # Save conversation to memory after streaming is complete
            if chat_id and full_response and not full_response.startswith(("Error:", "Cannot connect", "The AI service", "Sorry,")):
                self.save_conversation(chat_id, prompt, full_response)
                logger.info(f"Streaming LLM response generated for user {chat_id} in chat")
            elif chunk_count > 0:
                logger.info(f"Streaming completed with {chunk_count} chunks for chat {chat_id}")
                
        except Exception as e:
            logger.error(f"Streaming failed, falling back to regular response: {e}")
            # Fallback to regular response
            try:
                response = self.generate_response(prompt, chat_id, stop)
                yield response
            except Exception as fallback_error:
                logger.error(f"Fallback response also failed: {fallback_error}")
                yield "I'm sorry, I'm experiencing technical difficulties right now. Please try again later."
    
    def _build_message_context(self, prompt: str, chat_id: Optional[str]) -> List[Dict]:
        """Build message context including conversation history."""
        messages = []

        # Add system message first (only once)
        messages.append({
            "role": "system",
            "content": "you are Sahabat-ai a helpful AI Assistant from IOH"
        })

        # Add conversation context if available
        if chat_id and self.memory_manager:
            context_messages = self.memory_manager.get_conversation_context(chat_id)
            messages.extend(context_messages)

        # Add current user message
        messages.append({"role": "user", "content": prompt})

        return messages
    
    def save_conversation(self, chat_id: str, user_message: str, ai_response: str) -> None:
        """Save a conversation exchange to memory."""
        if self.memory_manager:
            self.memory_manager.add_message(chat_id, user_message, ai_response)
            logger.debug(f"Saved conversation to memory for chat {chat_id}")
    
    def get_conversation_history(self, chat_id: str, max_messages: int = 20) -> List[Dict]:
        """Get conversation history from memory."""
        if self.memory_manager:
            return self.memory_manager.get_conversation_context(chat_id, max_messages)
        return []
    
    def clear_conversation(self, chat_id: str) -> None:
        """Clear conversation history for a specific chat."""
        if self.memory_manager:
            self.memory_manager.clear_memory(chat_id)
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        if self.memory_manager:
            return self.memory_manager.get_memory_stats()
        return {"error": "No memory manager available"}


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global conversation memory manager instance
GLOBAL_MEMORY_MANAGER = ConversationMemoryManager()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_llm_instance(config: Optional[Dict] = None) -> SahabatLLM:
    """
    Create a SahabatLLM instance with default or provided configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SahabatLLM instance
    """
    if config is None:
        config = ENV_CONFIG
    
    return SahabatLLM(
        endpoint_url=config['endpoint_url'],
        api_key=config['api_key'],
        model_name=config['model_name'],
        temperature=config['temperature'],
        max_token=config['max_token'],
        memory_manager=GLOBAL_MEMORY_MANAGER
    )


def health_check() -> Dict:
    """
    Perform a health check of the LLM service.
    
    Returns:
        Health check results
    """
    try:
        llm = create_llm_instance()
        test_response = llm.generate_response("Hello, this is a health check.")
        
        return {
            "status": "healthy",
            "endpoint": CONFIG.endpoint_url,
            "model": CONFIG.model_name,
            "test_response_length": len(test_response),
            "memory_stats": llm.get_memory_stats()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "endpoint": CONFIG.endpoint_url if 'CONFIG' in globals() else "unknown"
        }