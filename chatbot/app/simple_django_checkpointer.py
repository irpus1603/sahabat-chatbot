"""
Simple Django-based checkpointer for LangGraph compatibility.

This provides basic database persistence for conversation state
without relying on complex LangGraph checkpoint interfaces.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from django.utils import timezone
from .models import LangGraphCheckpoint, LangGraphConversationState, Chat

logger = logging.getLogger(__name__)


class SimpleDjangoCheckpointer:
    """
    Simple Django-based checkpointer for basic conversation persistence.
    
    This stores conversation state in Django database tables without
    implementing the full LangGraph checkpoint interface.
    """
    
    def __init__(self):
        """Initialize the simple Django checkpointer."""
        self.logger = logger
    
    def save_conversation_state(self, thread_id: str, messages: List[Any], metadata: Dict[str, Any] = None) -> bool:
        """
        Save conversation state to database.
        
        Args:
            thread_id: Thread identifier
            messages: List of conversation messages
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize messages
            serialized_messages = self._serialize_messages(messages)
            
            # Create checkpoint
            checkpoint_id = str(uuid.uuid4())
            checkpoint_data = {
                'messages': serialized_messages,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
            }
            
            # Save to database
            checkpoint_obj, created = LangGraphCheckpoint.objects.update_or_create(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                defaults={
                    'checkpoint_data': checkpoint_data,
                    'metadata': metadata or {},
                }
            )
            
            # Update conversation state
            self._update_conversation_state(thread_id, checkpoint_id, messages)
            
            self.logger.debug(f"Saved conversation state for thread {thread_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation state for thread {thread_id}: {e}")
            return False
    
    def load_conversation_state(self, thread_id: str) -> Optional[List[Any]]:
        """
        Load conversation state from database.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            List of messages or None if not found
        """
        try:
            # Get latest checkpoint
            checkpoint_obj = LangGraphCheckpoint.objects.filter(
                thread_id=thread_id
            ).order_by('-created_at').first()
            
            if not checkpoint_obj:
                return None
            
            # Deserialize messages
            checkpoint_data = checkpoint_obj.checkpoint_data
            serialized_messages = checkpoint_data.get('messages', [])
            
            messages = self._deserialize_messages(serialized_messages)
            self.logger.debug(f"Loaded {len(messages)} messages for thread {thread_id}")
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation state for thread {thread_id}: {e}")
            return None
    
    def clear_conversation_state(self, thread_id: str) -> bool:
        """
        Clear conversation state for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete checkpoints
            deleted_checkpoints = LangGraphCheckpoint.objects.filter(thread_id=thread_id).delete()
            
            # Delete conversation state
            deleted_state = LangGraphConversationState.objects.filter(thread_id=thread_id).delete()
            
            self.logger.info(f"Cleared conversation state for thread {thread_id}: {deleted_checkpoints[0]} checkpoints, {deleted_state[0]} states")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear conversation state for thread {thread_id}: {e}")
            return False
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            total_checkpoints = LangGraphCheckpoint.objects.count()
            total_conversations = LangGraphConversationState.objects.count()
            active_conversations = LangGraphConversationState.objects.filter(is_active=True).count()
            
            return {
                "total_checkpoints": total_checkpoints,
                "total_conversations": total_conversations,
                "active_conversations": active_conversations,
                "storage_type": "django_database",
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation stats: {e}")
            return {"error": str(e)}
    
    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize messages for database storage.
        
        Args:
            messages: List of message objects
            
        Returns:
            List of serialized message dictionaries
        """
        serialized = []
        
        for msg in messages:
            try:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    # LangChain message format
                    serialized.append({
                        'type': msg.type,
                        'content': str(msg.content),
                        'additional_kwargs': getattr(msg, 'additional_kwargs', {}),
                    })
                elif isinstance(msg, dict):
                    # Dictionary format
                    serialized.append(msg)
                else:
                    # Fallback - convert to string
                    serialized.append({
                        'type': 'unknown',
                        'content': str(msg),
                        'additional_kwargs': {},
                    })
            except Exception as e:
                self.logger.warning(f"Failed to serialize message: {e}")
                serialized.append({
                    'type': 'error',
                    'content': str(msg),
                    'additional_kwargs': {},
                })
        
        return serialized
    
    def _deserialize_messages(self, serialized_messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Deserialize messages from database storage.
        
        Args:
            serialized_messages: List of serialized message dictionaries
            
        Returns:
            List of reconstructed message objects
        """
        messages = []
        
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            for msg_data in serialized_messages:
                try:
                    msg_type = msg_data.get('type', 'unknown')
                    content = msg_data.get('content', '')
                    
                    if msg_type == 'human':
                        messages.append(HumanMessage(content=content))
                    elif msg_type == 'ai':
                        messages.append(AIMessage(content=content))
                    else:
                        # Fallback - create basic message
                        messages.append(HumanMessage(content=content))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize message: {e}")
                    # Create fallback message
                    messages.append(HumanMessage(content=str(msg_data)))
        
        except ImportError:
            # Fallback if LangChain not available
            messages = [{'content': msg.get('content', '')} for msg in serialized_messages]
        
        return messages
    
    def _update_conversation_state(self, thread_id: str, checkpoint_id: str, messages: List[Any]) -> None:
        """
        Update conversation state tracking.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Checkpoint identifier
            messages: List of messages
        """
        try:
            # Try to link with Django Chat model
            try:
                chat_id = int(thread_id)
                chat = Chat.objects.get(id=chat_id)
            except (ValueError, Chat.DoesNotExist):
                chat = None
            
            # Update or create conversation state
            conversation_state, created = LangGraphConversationState.objects.update_or_create(
                thread_id=thread_id,
                defaults={
                    'chat': chat,
                    'current_checkpoint_id': checkpoint_id,
                    'message_count': len(messages),
                    'last_activity': timezone.now(),
                    'is_active': True,
                }
            )
            
            if created:
                self.logger.debug(f"Created new conversation state for thread {thread_id}")
            else:
                self.logger.debug(f"Updated conversation state for thread {thread_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update conversation state: {e}")


# Global instance
_simple_checkpointer = None


def get_simple_django_checkpointer() -> SimpleDjangoCheckpointer:
    """
    Get or create a simple Django checkpointer instance.
    
    Returns:
        SimpleDjangoCheckpointer instance
    """
    global _simple_checkpointer
    if _simple_checkpointer is None:
        _simple_checkpointer = SimpleDjangoCheckpointer()
    return _simple_checkpointer