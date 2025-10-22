from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chats')
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.title or f"Chat {self.id}"

class Message(models.Model):
    SENDER_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.sender}: {self.content[:50]}..."


class LangGraphCheckpoint(models.Model):
    """
    Model to store LangGraph checkpoints for persistent conversation state.
    This replaces MemorySaver with database-backed storage.
    """
    thread_id = models.CharField(max_length=255, db_index=True, help_text="Chat session identifier")
    checkpoint_id = models.CharField(max_length=255, help_text="Unique checkpoint identifier")
    parent_checkpoint_id = models.CharField(max_length=255, null=True, blank=True, help_text="Parent checkpoint for versioning")
    
    # Checkpoint data
    checkpoint_data = models.JSONField(help_text="Serialized checkpoint state")
    metadata = models.JSONField(default=dict, help_text="Checkpoint metadata")
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['thread_id', '-created_at']),
            models.Index(fields=['checkpoint_id']),
        ]
        unique_together = ['thread_id', 'checkpoint_id']

    def __str__(self):
        return f"Checkpoint {self.checkpoint_id} for thread {self.thread_id}"


class LangGraphConversationState(models.Model):
    """
    Model to store current conversation state for each chat session.
    Links Django Chat model with LangGraph state management.
    """
    chat = models.OneToOneField(Chat, on_delete=models.CASCADE, related_name='langgraph_state', null=True, blank=True)
    thread_id = models.CharField(max_length=255, unique=True, help_text="LangGraph thread identifier")
    
    # Current state
    current_checkpoint_id = models.CharField(max_length=255, null=True, blank=True)
    message_count = models.PositiveIntegerField(default=0, help_text="Number of messages in conversation")
    
    # Conversation context
    context_summary = models.TextField(blank=True, help_text="Summary of conversation context")
    user_context = models.JSONField(default=dict, help_text="User-specific context data")
    
    # State management
    is_active = models.BooleanField(default=True, help_text="Whether conversation is currently active")
    last_activity = models.DateTimeField(default=timezone.now)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_activity']
        indexes = [
            models.Index(fields=['thread_id']),
            models.Index(fields=['is_active', '-last_activity']),
        ]

    def __str__(self):
        return f"LangGraph state for chat {self.chat_id} (thread: {self.thread_id})"
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = timezone.now()
        self.save(update_fields=['last_activity'])
    
    def increment_message_count(self):
        """Increment message count."""
        self.message_count += 1
        self.save(update_fields=['message_count'])


class LangGraphMemoryStats(models.Model):
    """
    Model to track memory usage statistics for monitoring and optimization.
    """
    date = models.DateField(default=timezone.now, unique=True)
    
    # Statistics
    total_conversations = models.PositiveIntegerField(default=0)
    total_checkpoints = models.PositiveIntegerField(default=0)
    active_conversations = models.PositiveIntegerField(default=0)
    
    # Memory usage
    avg_checkpoints_per_conversation = models.FloatField(default=0.0)
    total_memory_mb = models.FloatField(default=0.0)
    
    # Performance metrics
    avg_response_time_ms = models.FloatField(default=0.0)
    total_api_calls = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Memory stats for {self.date}"
