from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class RAGDocument(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    source = models.CharField(max_length=500, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    chunk_count = models.IntegerField(default=0)
    embedding_status= models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing','Processing'),
            ('completed', 'Completed'),
            ('failed','Failed')
        ],
        default='pending'
    )

    class Meta:
        ordering = ['-created_at']


    def __str__(self):
        return self.title


class RAGQuery(models.Model):
    """Model to track RAG queries for analytics"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.TextField()
    results_count = models.IntegerField()
    avg_distance = models.FloatField()
    processing_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Query: {self.query[:50]}..."

class ChatMessage(models.Model):
    """Model to store chat messages for RAG context"""
    conversation = models.ForeignKey('app.Chat', on_delete=models.CASCADE, related_name='rag_messages')
    message_type = models.CharField(
        max_length=10,
        choices=[
            ('user', 'User'),
            ('assistant', 'Assistant')
        ]
    )
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."

class RAGEmbedding(models.Model):
    """Model to store document embeddings for vector search"""
    document = models.ForeignKey(RAGDocument, on_delete=models.CASCADE, related_name='embeddings')
    chunk_index = models.IntegerField()
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    embedding = models.BinaryField()  # Store embedding as binary data
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['document_id', 'chunk_index']
        indexes = [
            models.Index(fields=['document']),
            models.Index(fields=['document', 'chunk_index']),
        ]
    
    def __str__(self):
        return f"Embedding {self.chunk_index} for {self.document.title}"

class ChatRAGContext(models.Model):
    """Model to store RAG context used in chat responses"""
    chat_message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='rag_contexts')
    embedding = models.ForeignKey(RAGEmbedding, on_delete=models.CASCADE, related_name='rag_contexts', null=True)
    relevance_score = models.FloatField()
    used_in_response = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-relevance_score']
    
    def __str__(self):
        return f"Context for {self.chat_message.id}: {self.relevance_score}"
