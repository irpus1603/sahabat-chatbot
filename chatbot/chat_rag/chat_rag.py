import numpy as np
import json
import time
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from django.conf import settings
from django.db import models, connection

from .models import RAGDocument, ChatMessage, ChatRAGContext, RAGQuery, RAGEmbedding

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            settings.RAG_CONFIG['EMBEDDING_MODEL']
        )
        self.chunk_size = settings.RAG_CONFIG['CHUNK_SIZE']
        self.chunk_overlap = settings.RAG_CONFIG['CHUNK_OVERLAP']
        self.top_k = settings.RAG_CONFIG['TOP_K_RESULTS']
        self._setup_vector_db()
    
    def _setup_vector_db(self):
        """Setup vector database using Django models"""
        try:
            # Ensure Django models are created
            logger.info("Using Django models for RAG embeddings storage")
            self.vector_support = False  # Using manual similarity search
            
        except Exception as e:
            logger.error("Error setting up vector database: %s", str(e))
            self.vector_support = False
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                if boundary > start + self.chunk_size // 2:
                    chunk = chunk[:boundary + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            if end >= len(text):
                break
        
        return chunks
    
    def embed_document(self, document: RAGDocument) -> bool:
        """Embed a document and store in vector database"""
        try:
            document.embedding_status = 'processing'
            document.save()
            
            # Chunk the document
            chunks = self.chunk_text(document.content)
            document.chunk_count = len(chunks)
            document.save()
            
            with connection.cursor() as cursor:
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk)
                    embedding_bytes = np.array(embedding).tobytes()
                    
                    # Store embedding using Django model
                    metadata = {
                        'document_id': document.id,
                        'chunk_index': i,
                        'source': document.source,
                        'title': document.title,
                        **document.metadata
                    }
                    
                    # Create RAG embedding record
                    rag_embedding = RAGEmbedding.objects.create(
                        document=document,
                        chunk_index=i,
                        content=chunk,
                        metadata=metadata,
                        embedding=embedding_bytes
                    )
            
            document.embedding_status = 'completed'
            document.save()
            
            logger.info("Successfully embedded document %s with %d chunks", document.id, len(chunks))
            return True
            
        except Exception as e:
            logger.error("Error embedding document %s: %s", document.id, str(e))
            document.embedding_status = 'failed'
            document.save()
            return False
    
    def search_similar(self, query: str, top_k: Optional[int] = None, 
                      user_id: Optional[int] = None) -> List[Dict]:
        """Search for similar content using vector similarity"""
        if top_k is None:
            top_k = self.top_k
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            results = []
            
            # Use Django ORM to get embeddings and compute similarity
            candidate_results = []
            
            # Get all embeddings using Django ORM (batch processing)
            embeddings_queryset = RAGEmbedding.objects.all().order_by('id')
            
            for embedding_obj in embeddings_queryset.iterator(chunk_size=100):
                try:
                    # Convert embedding bytes back to numpy array
                    stored_embedding = np.frombuffer(embedding_obj.embedding, dtype=np.float32)
                    
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    # Convert similarity to distance (lower is better)
                    distance = 1.0 - similarity
                    relevance_score = max(0.0, similarity)  # Ensure non-negative
                    
                    # Only include results with reasonable relevance
                    if relevance_score > 0.1:  # Threshold for relevance
                        candidate_results.append({
                            'id': embedding_obj.id,
                            'document_id': embedding_obj.document.id,
                            'chunk_index': embedding_obj.chunk_index,
                            'content': embedding_obj.content,
                            'metadata': embedding_obj.metadata,
                            'distance': distance,
                            'relevance_score': relevance_score
                        })
                except Exception as e:
                    logger.warning(f"Error computing similarity for embedding {embedding_obj.id}: {e}")
                    continue
            
            # Sort by relevance and take top results
            candidate_results.sort(key=lambda x: x['distance'])
            results = candidate_results[:top_k]
                    
            # If no semantic matches, fall back to text search
            if not results:
                query_terms = query.lower().split()
                
                # Use Django ORM for text search
                text_search_embeddings = RAGEmbedding.objects.filter(
                    models.Q(content__icontains=query) | 
                    models.Q(content__icontains=query.lower())
                )[:top_k]
                
                for embedding_obj in text_search_embeddings:
                    content = embedding_obj.content.lower()
                    
                    # Calculate text-based relevance score
                    text_score = 0.0
                    for term in query_terms:
                        if term in content:
                            text_score += 0.3  # Base score for term match
                            # Bonus for exact phrase match
                            if query.lower() in content:
                                text_score += 0.2
                    
                    text_score = min(text_score, 1.0)  # Cap at 1.0
                    
                    if text_score > 0.1:  # Only include if some relevance
                        results.append({
                            'id': embedding_obj.id,
                            'document_id': embedding_obj.document.id,
                            'chunk_index': embedding_obj.chunk_index,
                            'content': embedding_obj.content,
                            'metadata': embedding_obj.metadata,
                            'distance': 1.0 - text_score,
                            'relevance_score': text_score
                        })
                
            processing_time = time.time() - start_time
            avg_distance = sum(r['distance'] for r in results) / len(results) if results else 0
                
            # Log query for analytics
            if user_id:
                from django.contrib.auth.models import User
                try:
                    user = User.objects.get(id=user_id)
                    RAGQuery.objects.create(
                        user=user,
                        query=query,
                        results_count=len(results),
                        avg_distance=avg_distance,
                        processing_time=processing_time
                    )
                except User.DoesNotExist:
                    pass
                
            return results
                
        except Exception as e:
            logger.error("Error searching similar content: %s", str(e))
            return []
    
    def get_chat_context(self, query: str, user_id: int, 
                        conversation_id: Optional[int] = None) -> List[Dict]:
        """Get context including chat history and RAG results"""
        
        # Get RAG context
        rag_context = self.search_similar(query, user_id=user_id)
        
        # Get relevant chat history
        chat_messages = ChatMessage.objects.filter(
            conversation__user_id=user_id,
            message_type='user'
        )
        
        if conversation_id:
            chat_messages = chat_messages.filter(conversation_id=conversation_id)
        
        # Simple keyword matching for chat history
        relevant_messages = chat_messages.filter(
            content__icontains=query.split()[0] if query.split() else query
        )[:3]
        
        # Add chat context
        for msg in relevant_messages:
            rag_context.append({
                'id': f'chat_{msg.id}',
                'content': msg.content,
                'metadata': {
                    'source': 'chat_history',
                    'timestamp': msg.timestamp.isoformat(),
                    'conversation_id': msg.conversation.id
                },
                'distance': 0.3,  # Fixed relevance for chat
                'relevance_score': 0.7
            })
        
        # Sort by relevance
        return sorted(rag_context, key=lambda x: x['distance'])[:self.top_k]
    
    def delete_document_embeddings(self, document_id: int):
        """Delete all embeddings for a document"""
        try:
            # Delete using Django ORM
            embeddings_deleted = RAGEmbedding.objects.filter(document_id=document_id).delete()
            
            logger.info("Deleted %s embeddings for document %s", 
                       embeddings_deleted[0], document_id)
            return True
            
        except Exception as e:
            logger.error("Error deleting embeddings for document %s: %s", document_id, str(e))
            return False
    
    def get_analytics(self, user_id: Optional[int] = None) -> Dict:
        """Get RAG system analytics"""
        try:
            base_query = RAGQuery.objects.all()
            if user_id:
                base_query = base_query.filter(user_id=user_id)
            
            total_documents = RAGDocument.objects.filter(is_active=True).count()
            total_chunks = RAGEmbedding.objects.count()
            total_queries = base_query.count()
            
            avg_processing_time = base_query.aggregate(
                avg_time=models.Avg('processing_time')
            )['avg_time'] or 0
            
            avg_results = base_query.aggregate(
                avg_results=models.Avg('results_count')
            )['avg_results'] or 0
            
            return {
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'total_queries': total_queries,
                'avg_processing_time': round(avg_processing_time, 3),
                'avg_results_per_query': round(avg_results, 2),
                'embedding_status': {
                    status: count for status, count in 
                    RAGDocument.objects.values_list('embedding_status')
                    .annotate(count=models.Count('id'))
                }
            }
            
        except Exception as e:
            logger.error("Error getting analytics: %s", str(e))
            return {}

# Global RAG service instance
rag_service = RAGService()