"""
Django-based LangGraph Checkpointer

This module provides a custom checkpointer that stores LangGraph state
in Django database tables instead of using MemorySaver.
"""

import json
import uuid
from typing import Dict, Any, Optional, List, Iterator
from datetime import datetime

from django.utils import timezone
from langchain_core.runnables import RunnableConfig
try:
    from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
except ImportError:
    # Fallback for different LangGraph versions
    BaseCheckpointSaver = object
    Checkpoint = dict
    CheckpointMetadata = dict

from .models import LangGraphCheckpoint, LangGraphConversationState, Chat


class DjangoCheckpointSaver(BaseCheckpointSaver):
    """
    Django-based checkpoint saver for LangGraph.
    
    Stores checkpoints in Django database tables for true persistence
    across application restarts and scaling.
    """
    
    def __init__(self):
        """Initialize the Django checkpoint saver."""
        super().__init__()
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> None:
        """
        Save a checkpoint to the database.
        
        Args:
            config: Configuration containing thread_id
            checkpoint: The checkpoint data to save
            metadata: Checkpoint metadata
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = str(uuid.uuid4())
            
            # Serialize checkpoint data
            checkpoint_data = self._serialize_checkpoint(checkpoint)
            metadata_data = self._serialize_metadata(metadata)
            
            # Create or update checkpoint
            checkpoint_obj, created = LangGraphCheckpoint.objects.update_or_create(
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                defaults={
                    'checkpoint_data': checkpoint_data,
                    'metadata': metadata_data,
                    'parent_checkpoint_id': metadata.get('parent_checkpoint_id'),
                }
            )
            
            # Update conversation state
            self._update_conversation_state(thread_id, checkpoint_id, checkpoint)
            
        except Exception as e:
            # Log error but don't break the conversation
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save checkpoint for thread {thread_id}: {e}")
    
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """
        Retrieve the latest checkpoint for a thread.
        
        Args:
            config: Configuration containing thread_id
            
        Returns:
            Latest checkpoint or None if not found
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            
            # Get latest checkpoint
            checkpoint_obj = LangGraphCheckpoint.objects.filter(
                thread_id=thread_id
            ).order_by('-created_at').first()
            
            if not checkpoint_obj:
                return None
            
            # Deserialize checkpoint
            return self._deserialize_checkpoint(checkpoint_obj.checkpoint_data)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get checkpoint for thread {thread_id}: {e}")
            return None
    
    def list(self, config: RunnableConfig, limit: int = 10) -> List[CheckpointMetadata]:
        """
        List recent checkpoints for a thread.
        
        Args:
            config: Configuration containing thread_id
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            
            # Get recent checkpoints
            checkpoints = LangGraphCheckpoint.objects.filter(
                thread_id=thread_id
            ).order_by('-created_at')[:limit]
            
            # Return metadata
            metadata_list = []
            for checkpoint in checkpoints:
                metadata = {
                    'checkpoint_id': checkpoint.checkpoint_id,
                    'thread_id': checkpoint.thread_id,
                    'created_at': checkpoint.created_at.isoformat(),
                    'parent_checkpoint_id': checkpoint.parent_checkpoint_id,
                }
                metadata.update(checkpoint.metadata)
                metadata_list.append(metadata)
            
            return metadata_list
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to list checkpoints for thread {thread_id}: {e}")
            return []
    
    def delete(self, config: RunnableConfig) -> None:
        """
        Delete all checkpoints for a thread.
        
        Args:
            config: Configuration containing thread_id
        """
        try:
            thread_id = config["configurable"]["thread_id"]
            
            # Delete all checkpoints for this thread
            LangGraphCheckpoint.objects.filter(thread_id=thread_id).delete()
            
            # Delete conversation state
            LangGraphConversationState.objects.filter(thread_id=thread_id).delete()
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to delete checkpoints for thread {thread_id}: {e}")
    
    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> Dict[str, Any]:
        """
        Serialize checkpoint data for database storage.
        
        Args:
            checkpoint: Checkpoint to serialize
            
        Returns:
            Serializable dictionary
        """
        try:
            # Convert checkpoint to JSON-serializable format
            if hasattr(checkpoint, 'channel_values'):
                # Extract message data
                messages = checkpoint.channel_values.get('messages', [])
                serialized_messages = []
                
                for msg in messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        serialized_messages.append({
                            'type': msg.type,
                            'content': msg.content,
                            'additional_kwargs': getattr(msg, 'additional_kwargs', {}),
                        })
                
                return {
                    'messages': serialized_messages,
                    'channel_values': checkpoint.channel_values,
                    'checkpoint_ns': getattr(checkpoint, 'checkpoint_ns', ''),
                    'checkpoint_id': getattr(checkpoint, 'checkpoint_id', ''),
                }
            else:
                # Fallback serialization
                return {'raw_checkpoint': str(checkpoint)}
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to serialize checkpoint: {e}")
            return {'raw_checkpoint': str(checkpoint)}
    
    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> Checkpoint:
        """
        Deserialize checkpoint data from database.
        
        Args:
            data: Serialized checkpoint data
            
        Returns:
            Reconstructed checkpoint
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Reconstruct messages
            messages = []
            for msg_data in data.get('messages', []):
                if msg_data['type'] == 'human':
                    messages.append(HumanMessage(content=msg_data['content']))
                elif msg_data['type'] == 'ai':
                    messages.append(AIMessage(content=msg_data['content']))
            
            # Create basic checkpoint structure
            # Note: This is a simplified reconstruction
            # In production, you'd want more complete deserialization
            checkpoint = type('Checkpoint', (), {
                'channel_values': {'messages': messages},
                'checkpoint_ns': data.get('checkpoint_ns', ''),
                'checkpoint_id': data.get('checkpoint_id', ''),
            })()
            
            return checkpoint
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to deserialize checkpoint: {e}")
            # Return empty checkpoint
            return type('Checkpoint', (), {'channel_values': {'messages': []}})()
    
    def _serialize_metadata(self, metadata: CheckpointMetadata) -> Dict[str, Any]:
        """
        Serialize checkpoint metadata.
        
        Args:
            metadata: Metadata to serialize
            
        Returns:
            Serializable dictionary
        """
        try:
            if isinstance(metadata, dict):
                return metadata
            else:
                # Convert metadata object to dict
                return {
                    'step': getattr(metadata, 'step', 0),
                    'writes': getattr(metadata, 'writes', {}),
                    'source': getattr(metadata, 'source', 'checkpoint'),
                }
        except Exception:
            return {}
    
    def _update_conversation_state(self, thread_id: str, checkpoint_id: str, checkpoint: Checkpoint) -> None:
        """
        Update the conversation state tracking.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Checkpoint identifier
            checkpoint: Checkpoint data
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
                    'last_activity': timezone.now(),
                    'is_active': True,
                }
            )
            
            # Count messages in checkpoint
            if hasattr(checkpoint, 'channel_values'):
                messages = checkpoint.channel_values.get('messages', [])
                conversation_state.message_count = len(messages)
                conversation_state.save(update_fields=['message_count'])
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update conversation state: {e}")


# Custom checkpointer factory
def create_django_checkpointer() -> DjangoCheckpointSaver:
    """
    Create a Django-based checkpointer instance.
    
    Returns:
        DjangoCheckpointSaver instance
    """
    return DjangoCheckpointSaver()