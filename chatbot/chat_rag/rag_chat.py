"""
RAG Chat Integration Module

This module integrates the existing LLM endpoint with RAG functionality,
combining document retrieval with chat generation.
"""

import logging
from typing import Dict, List, Optional, Generator
from django.conf import settings

# Import existing LLM functionality
from app.chat import create_llm_instance
from .chat_rag import rag_service

logger = logging.getLogger(__name__)

class RAGChatService:
    """Service for handling RAG-enhanced chat interactions"""
    
    def __init__(self):
        self.llm = create_llm_instance()
        self.rag_service = rag_service
    
    def generate_rag_response(self, query: str, chat_id: str, user_id: int) -> str:
        """Generate response using RAG context and LLM"""
        try:
            # Log original query
            logger.info(f"Original query for chat {chat_id}: {query[:100]}...")
            
            # Get chat history for context
            chat_history = self._get_chat_history(chat_id, limit=5)
            logger.info(f"Retrieved {len(chat_history)} chat history messages for context")
            
            # Build enhanced query with chat history
            enhanced_query = self._build_contextual_query(query, chat_history)
            
            # Log enhanced query to see if history is included
            if enhanced_query != query:
                logger.info(f"Enhanced query with history for chat {chat_id}: {enhanced_query[:200]}...")
            else:
                logger.info(f"No chat history context added - using original query")
            
            # Get RAG context using enhanced query
            rag_context = self.rag_service.search_similar(enhanced_query, user_id=user_id)
            logger.info(f"Found {len(rag_context)} RAG context results")
            
            # Log RAG context details
            if rag_context:
                for i, ctx in enumerate(rag_context[:3]):  # Log top 3 contexts
                    relevance = ctx.get('relevance_score', 0)
                    content_preview = ctx.get('content', '')[:80] + "..." if len(ctx.get('content', '')) > 80 else ctx.get('content', '')
                    source = ctx.get('metadata', {}).get('source', 'Unknown')
                    logger.debug(f"RAG Context {i+1}: Score={relevance:.3f}, Source={source}, Content={content_preview}")
            else:
                logger.info("No RAG contexts found for the query")
            
            # Build enhanced prompt with context
            enhanced_prompt = self._build_rag_prompt(query, rag_context, chat_history)
            logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
            
            # Generate response using existing LLM
            response = self.llm.generate_response(enhanced_prompt, chat_id=chat_id)
            
            # Log detailed response information
            logger.info(f"RAG response generated for user {user_id}, chat {chat_id}")
            logger.info(f"Response length: {len(response)} characters")
            logger.info(f"Response preview: {response[:150]}...")
            
            # Log RAG context usage summary
            high_relevance_contexts = [ctx for ctx in rag_context if ctx.get('relevance_score', 0) > 0.15]
            logger.info(f"Used {len(high_relevance_contexts)}/{len(rag_context)} RAG contexts with relevance > 0.15")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            # Fallback to regular LLM response
            return self.llm.generate_response(query, chat_id=chat_id)

    def generate_rag_response_streaming(self, query: str, chat_id: str, user_id: int) -> Generator[str, None, None]:
        """Generate streaming response using RAG context and LLM"""
        try:
            # Log original query
            logger.info(f"Streaming RAG query for chat {chat_id}: {query[:100]}...")

            # Get chat history for context
            chat_history = self._get_chat_history(chat_id, limit=5)
            logger.info(f"Retrieved {len(chat_history)} chat history messages for context")

            # Build enhanced query with chat history
            enhanced_query = self._build_contextual_query(query, chat_history)

            # Log enhanced query
            if enhanced_query != query:
                logger.info(f"Enhanced query with history for chat {chat_id}: {enhanced_query[:200]}...")

            # Get RAG context using enhanced query
            rag_context = self.rag_service.search_similar(enhanced_query, user_id=user_id)
            logger.info(f"Found {len(rag_context)} RAG context results")

            # Log RAG context details
            if rag_context:
                for i, ctx in enumerate(rag_context[:3]):
                    relevance = ctx.get('relevance_score', 0)
                    content_preview = ctx.get('content', '')[:80] + "..." if len(ctx.get('content', '')) > 80 else ctx.get('content', '')
                    source = ctx.get('metadata', {}).get('source', 'Unknown')
                    logger.debug(f"RAG Context {i+1}: Score={relevance:.3f}, Source={source}, Content={content_preview}")

            # Build enhanced prompt with context
            enhanced_prompt = self._build_rag_prompt(query, rag_context, chat_history)
            logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)} characters")

            # Generate streaming response using existing LLM
            full_response = ""
            chunk_count = 0

            for chunk in self.llm.generate_streaming_response(enhanced_prompt, chat_id=chat_id):
                full_response += chunk
                chunk_count += 1
                yield chunk

            # Log response summary
            logger.info(f"Streaming RAG response generated for user {user_id}, chat {chat_id}")
            logger.info(f"Response: {chunk_count} chunks, {len(full_response)} characters")

            # Log RAG context usage summary
            high_relevance_contexts = [ctx for ctx in rag_context if ctx.get('relevance_score', 0) > 0.15]
            logger.info(f"Used {len(high_relevance_contexts)}/{len(rag_context)} RAG contexts with relevance > 0.15")

        except Exception as e:
            logger.error(f"Error generating streaming RAG response: {str(e)}")
            # Fallback to regular streaming LLM response
            for chunk in self.llm.generate_streaming_response(query, chat_id=chat_id):
                yield chunk

    def _get_chat_history(self, chat_id: str, limit: int = 8) -> List[Dict]:
        """Retrieve recent chat history for context"""
        try:
            from app.models import Chat, Message
            
            chat = Chat.objects.get(id=chat_id)
            messages = Message.objects.filter(chat=chat).order_by('-created_at')[:limit*2]
            
            logger.debug(f"Raw messages count for chat {chat_id}: {messages.count()}")
            
            # Format messages for context
            history = []
            for message in reversed(messages):
                history.append({
                    'sender': message.sender,
                    'content': message.content,
                    'timestamp': message.created_at
                })
            
            logger.debug(f"Formatted chat history for chat {chat_id}: {len(history)} messages")
            if history:
                logger.debug(f"Last message: {history[-1]['sender']}: {history[-1]['content'][:50]}...")
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def _build_contextual_query(self, query: str, chat_history: List[Dict]) -> str:
        """Build enhanced query with chat history context"""
        if not chat_history:
            logger.debug("No chat history available for contextual query")
            return query
        
        # Get recent context (last 2-3 messages)
        recent_context = []
        for msg in chat_history[-4:]:  # Last 4 messages for context
            if msg['sender'] == 'user':
                recent_context.append(f"Previous question: {msg['content']}")
            elif msg['sender'] == 'assistant':
                # Include key points from assistant responses
                content = msg['content'][:200]  # Truncate long responses
                recent_context.append(f"Previous answer context: {content}")
        
        logger.debug(f"Recent context items: {len(recent_context)}")
        
        if recent_context:
            contextual_query = f"{' '.join(recent_context)} Current question: {query}"
            logger.debug(f"Built contextual query with {len(recent_context)} context items")
            return contextual_query
        
        logger.debug("No relevant context found in chat history")
        return query
    
    def _build_rag_prompt(self, query: str, rag_context: List[Dict], chat_history: Optional[List[Dict]] = None) -> str:
        """Build enhanced prompt with RAG context and chat history"""
        if not rag_context:
            return query
        
        # Build context section
        context_sections = []
        for idx, context in enumerate(rag_context[:10]):  # Use top 10 results
            relevance = context.get('relevance_score', 0)
            content = context.get('content', '').strip()
            source = context.get('metadata', {}).get('source', 'Unknown')
            
            if relevance > 0.15 and content:  # Lower threshold for better context inclusion
                context_sections.append(f"Context {idx + 1} (Relevance: {relevance:.2f}):\n{content}\nSource: {source}\n")
        
        if not context_sections:
            return query
        
        # Build conversation context if available
        conversation_context = ""
        if chat_history:
            recent_messages = chat_history[-4:]  # Last 4 messages
            if recent_messages:
                conversation_context = "\n\nRecent Conversation Context:\n"
                for msg in recent_messages:
                    role = "User" if msg['sender'] == 'user' else "Assistant"
                    content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                    conversation_context += f"{role}: {content}\n"
        
        # Build enhanced prompt
        enhanced_prompt = f"""Based on the following context information and conversation history, please answer the user's question. 
        If the context doesn't contain relevant information, you can use your general knowledge but mention that you're not 
        finding specific information in the provided documents. Consider the conversation context to provide continuity.

        Document Context Information:
        {chr(10).join(context_sections)}{conversation_context}

        Current User Question: {query}

        Please provide a comprehensive answer using the context information when relevant, maintain conversation continuity, and cite the sources when possible."""
        
        return enhanced_prompt
    
    def search_documents(self, query: str, user_id: int, top_k: int = 10) -> List[Dict]:
        """Search documents and return formatted results"""
        try:
            results = self.rag_service.search_similar(query, top_k=top_k, user_id=user_id)
            
            # Format results for display
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                    'source': result.get('metadata', {}).get('source', 'Unknown'),
                    'title': result.get('metadata', {}).get('title', 'Untitled'),
                    'relevance_score': result.get('relevance_score', 0),
                    'document_id': result.get('document_id'),
                    'chunk_index': result.get('chunk_index')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_chat_analytics(self, user_id: Optional[int] = None) -> Dict:
        """Get analytics for RAG chat usage"""
        try:
            # Get RAG analytics
            rag_analytics = self.rag_service.get_analytics(user_id=user_id)
            
            # Get LLM memory stats
            llm_stats = self.llm.get_memory_stats()
            
            return {
                'rag_analytics': rag_analytics,
                'llm_stats': llm_stats,
                'integration_status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting chat analytics: {str(e)}")
            return {
                'error': str(e),
                'integration_status': 'error'
            }
    
    def clear_chat_context(self, chat_id: str) -> bool:
        """Clear both RAG and LLM context for a chat"""
        try:
            # Clear LLM memory
            self.llm.clear_conversation(chat_id)
            
            # RAG context is query-based, so no specific clearing needed
            logger.info(f"Cleared chat context for chat {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing chat context: {str(e)}")
            return False

# Global RAG chat service instance
rag_chat_service = RAGChatService()