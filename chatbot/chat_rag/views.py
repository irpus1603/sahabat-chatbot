from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.template.loader import render_to_string
import os
import json
import logging
from pathlib import Path

from .models import RAGDocument, RAGQuery, RAGEmbedding
from .document_processor import document_processor
from .chat_rag import rag_service
from .rag_chat import rag_chat_service

logger = logging.getLogger(__name__)

def is_htmx_request(request):
    """Check if the request is from HTMX"""
    return request.headers.get('HX-Request') == 'true'

@login_required
def document_upload(request):
    """Document upload and management view"""
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('documents')
        uploaded_count = 0
        
        try:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                file_name = uploaded_file.name
                file_path = settings.RAG_CONFIG['DOCUMENT_FOLDER'] / file_name
                
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save file
                with open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                # Process the document
                doc = document_processor.process_file(file_path)
                if doc:
                    # Start embedding process
                    rag_service.embed_document(doc)
                    uploaded_count += 1
                    
            if is_htmx_request(request):
                return HttpResponse(f"""
                    <div class="alert alert-success alert-dismissible fade show">
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        Successfully uploaded {uploaded_count} documents
                    </div>
                """)
            else:
                return JsonResponse({
                    'success': True,
                    'uploaded_count': uploaded_count,
                    'message': f'Successfully uploaded {uploaded_count} documents'
                })
            
        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            if is_htmx_request(request):
                return HttpResponse(f"""
                    <div class="alert alert-danger alert-dismissible fade show">
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        Error uploading documents: {str(e)}
                    </div>
                """)
            else:
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })
    
    # GET request - show upload form or return partial for HTMX
    documents = RAGDocument.objects.filter(is_active=True).order_by('-created_at')
    
    # Check if this is an HTMX request for just the documents table
    if is_htmx_request(request) and request.headers.get('HX-Target') == '#documents-table':
        return render(request, 'chat_rag/partials/documents_table.html', {
            'documents': documents
        })
    
    # Get statistics
    total_documents = documents.count()
    total_chunks = RAGEmbedding.objects.count()
    completed_embeddings = documents.filter(embedding_status='completed').count()
    
    # Get folder scan statistics
    try:
        folder_stats = document_processor.scan_folder()
        # Debug: Log the folder stats being passed to template
        logger.info(f"Folder stats for template: {folder_stats}")
    except Exception as e:
        logger.error(f"Error getting folder stats: {e}")
        folder_stats = {
            'total_files': 0,
            'supported_files': 0,
            'by_extension': {},
            'folder_structure': [],
            'unprocessed_files': []
        }
    
    context = {
        'documents': documents,
        'document_folder': settings.RAG_CONFIG['DOCUMENT_FOLDER'],
        'stats': {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'completed_embeddings': completed_embeddings
        },
        'folder_stats': folder_stats
    }
    
    return render(request, 'chat_rag/document_upload.html', context)

@login_required
def process_folder(request):
    """Process all documents in the configured folder"""
    if request.method == 'POST':
        try:
            # Process documents in folder
            documents = document_processor.process_folder()
            processed_count = 0
            
            # Start embedding process for each document
            for doc in documents:
                if doc.embedding_status == 'pending':
                    rag_service.embed_document(doc)
                    processed_count += 1
            
            if is_htmx_request(request):
                return HttpResponse(f"""
                    <div class="alert alert-success alert-dismissible fade show">
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        Successfully processed {processed_count} documents
                    </div>
                """)
            else:
                return JsonResponse({
                    'success': True,
                    'processed_count': processed_count,
                    'message': f'Successfully processed {processed_count} documents'
                })
            
        except Exception as e:
            logger.error(f"Error processing folder: {str(e)}")
            if is_htmx_request(request):
                return HttpResponse(f"""
                    <div class="alert alert-danger alert-dismissible fade show">
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        Error processing folder: {str(e)}
                    </div>
                """)
            else:
                return JsonResponse({
                    'success': False,
                    'error': str(e)
                })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def reprocess_document(request, doc_id):
    """Reprocess a specific document"""
    if request.method == 'POST':
        try:
            doc = get_object_or_404(RAGDocument, id=doc_id, is_active=True)
            
            # Clear existing embeddings
            rag_service.delete_document_embeddings(doc_id)
            
            # Reset status
            doc.embedding_status = 'pending'
            doc.chunk_count = 0
            doc.save()
            
            # Start embedding process
            success = rag_service.embed_document(doc)
            
            if is_htmx_request(request):
                # Return updated documents table
                documents = RAGDocument.objects.filter(is_active=True).order_by('-created_at')
                return render(request, 'chat_rag/partials/documents_table.html', {
                    'documents': documents
                })
            else:
                return JsonResponse({
                    'success': success,
                    'message': 'Document reprocessed successfully' if success else 'Failed to reprocess document'
                })
            
        except Exception as e:
            logger.error(f"Error reprocessing document {doc_id}: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def delete_document(request, doc_id):
    """Delete a document and its embeddings"""
    if request.method == 'POST':
        try:
            doc = get_object_or_404(RAGDocument, id=doc_id, is_active=True)
            
            # Delete embeddings
            rag_service.delete_document_embeddings(doc_id)
            
            # Delete physical file if it exists
            if doc.source and doc.source.startswith('/'):
                try:
                    file_path = Path(doc.source)
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Deleted physical file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting physical file {doc.source}: {str(e)}")
            
            # Delete document from database
            doc.is_active = False
            doc.save()
            
            if is_htmx_request(request):
                # Return updated documents table
                documents = RAGDocument.objects.filter(is_active=True).order_by('-created_at')
                return render(request, 'chat_rag/partials/documents_table.html', {
                    'documents': documents
                })
            else:
                return JsonResponse({
                    'success': True,
                    'message': 'Document deleted successfully'
                })
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def document_search(request):
    """Search documents using RAG"""
    results = []
    query = ""
    
    if request.method == 'POST':
        query = request.POST.get('query', '').strip()
        
        if query:
            try:
                # Search documents
                results = rag_chat_service.search_documents(
                    query=query,
                    user_id=request.user.id,
                    top_k=10
                )
                
            except Exception as e:
                logger.error(f"Error searching documents: {str(e)}")
                messages.error(request, f"Error searching documents: {str(e)}")
    
    context = {
        'results': results,
        'query': query
    }
    
    return render(request, 'chat_rag/document_search.html', context)

@login_required
def rag_chat_stream_api(request):
    """Streaming API endpoint for RAG-enhanced chat"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_content = data.get('message', '').strip()
            chat_id = data.get('chat_id')

            if not message_content:
                return JsonResponse({'error': 'Message content is required'}, status=400)

            # Get or create chat
            from app.models import Chat, Message
            if chat_id:
                try:
                    chat_obj = Chat.objects.get(id=chat_id, user=request.user)
                except Chat.DoesNotExist:
                    return JsonResponse({'error': 'Chat not found'}, status=404)
            else:
                chat_obj = Chat.objects.create(
                    user=request.user,
                    title=message_content[:50]
                )

            # Save user message
            user_message = Message.objects.create(
                chat=chat_obj,
                sender='user',
                content=message_content
            )

            # Stream RAG response
            def stream_response():
                """Generator function for streaming RAG response"""
                try:
                    # Send initial acknowledgment
                    yield f"data: {json.dumps({'type': 'user_message', 'message_id': user_message.id, 'chat_id': chat_obj.id})}\n\n"

                    assistant_response = ""

                    # Generate streaming RAG response
                    for chunk in rag_chat_service.generate_rag_response_streaming(
                        query=message_content,
                        chat_id=str(chat_obj.id),
                        user_id=request.user.id
                    ):
                        assistant_response += chunk
                        # Send each chunk to the client
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                    logger.info(f"Streaming RAG response generated for user {request.user.username} in chat {chat_obj.id}")

                    # Save assistant response to database
                    assistant_message = Message.objects.create(
                        chat=chat_obj,
                        sender='assistant',
                        content=assistant_response
                    )

                    # Send completion signal
                    yield f"data: {json.dumps({'type': 'assistant_end', 'message_id': assistant_message.id})}\n\n"

                except Exception as e:
                    # Log the error and send error message
                    logger.error(f"Streaming RAG chat request failed for user {request.user.username}: {str(e)}")
                    error_response = "I'm sorry, I'm experiencing technical difficulties with document search right now. Please try again later."
                    yield f"data: {json.dumps({'type': 'content', 'content': error_response})}\n\n"

                    # Save error response to database
                    Message.objects.create(
                        chat=chat_obj,
                        sender='assistant',
                        content=error_response
                    )
                    yield f"data: {json.dumps({'type': 'assistant_end'})}\n\n"

                finally:
                    # Signal stream completion
                    yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

            # Return streaming response
            from django.http import StreamingHttpResponse
            return StreamingHttpResponse(
                stream_response(),
                content_type='text/event-stream'
            )

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"RAG chat streaming API error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def rag_chat_api(request):
    """API endpoint for RAG-enhanced chat (non-streaming)"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_content = data.get('message', '').strip()
            chat_id = data.get('chat_id')
            
            if not message_content:
                return JsonResponse({'error': 'Message content is required'}, status=400)
            
            # Get or create chat
            from app.models import Chat, Message
            if chat_id:
                try:
                    chat_obj = Chat.objects.get(id=chat_id, user=request.user)
                except Chat.DoesNotExist:
                    return JsonResponse({'error': 'Chat not found'}, status=404)
            else:
                chat_obj = Chat.objects.create(
                    user=request.user,
                    title=message_content[:50]
                )
            
            # Save user message
            user_message = Message.objects.create(
                chat=chat_obj,
                sender='user',
                content=message_content
            )
            
            # Generate RAG response
            try:
                assistant_response = rag_chat_service.generate_rag_response(
                    query=message_content,
                    chat_id=str(chat_obj.id),
                    user_id=request.user.id
                )
                
                logger.info(f"RAG response generated for user {request.user.username} in chat {chat_obj.id}")
                
            except Exception as e:
                logger.error(f"RAG chat failed for user {request.user.username}: {str(e)}")
                assistant_response = "I'm sorry, I'm experiencing technical difficulties with document search right now. Please try again later."
            
            # Save assistant response
            assistant_message = Message.objects.create(
                chat=chat_obj,
                sender='assistant',
                content=assistant_response
            )
            
            return JsonResponse({
                'success': True,
                'chat_id': chat_obj.id,
                'user_message': {
                    'id': user_message.id,
                    'content': user_message.content,
                    'sender': user_message.sender,
                    'timestamp': user_message.created_at.isoformat()
                },
                'assistant_message': {
                    'id': assistant_message.id,
                    'content': assistant_message.content,
                    'sender': assistant_message.sender,
                    'timestamp': assistant_message.created_at.isoformat()
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"RAG chat API error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def rag_analytics(request):
    """Get RAG system analytics"""
    if request.method == 'GET':
        try:
            analytics = rag_chat_service.get_chat_analytics(user_id=request.user.id)
            return JsonResponse({
                'success': True,
                'analytics': analytics
            })
            
        except Exception as e:
            logger.error(f"Error getting RAG analytics: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def scan_custom_folder(request):
    """Scan a custom folder for documents"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            folder_path = data.get('folder_path')
            recursive = data.get('recursive', True)
            auto_process = data.get('auto_process', False)
            
            if not folder_path:
                return JsonResponse({'error': 'Folder path is required'}, status=400)
            
            from pathlib import Path
            folder_path = Path(folder_path)
            
            if not folder_path.exists():
                return JsonResponse({'error': 'Folder does not exist'}, status=400)
            
            if not folder_path.is_dir():
                return JsonResponse({'error': 'Path is not a directory'}, status=400)
            
            # Scan the folder
            folder_stats = document_processor.scan_folder(folder_path)
            
            # Process unprocessed files if auto_process is enabled
            if auto_process and folder_stats['unprocessed_files']:
                processed_count = 0
                failed_count = 0
                
                for file_info in folder_stats['unprocessed_files']:
                    try:
                        file_path = Path(file_info['full_path'])
                        doc = document_processor.process_file(file_path)
                        if doc:
                            success = rag_service.embed_document(doc)
                            if success:
                                processed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"Error auto-processing {file_info['path']}: {str(e)}")
                        failed_count += 1
                
                # Update folder stats after processing
                folder_stats = document_processor.scan_folder(folder_path)
                
                return JsonResponse({
                    'success': True,
                    'folder_stats': folder_stats,
                    'auto_processed': True,
                    'processed_count': processed_count,
                    'failed_count': failed_count
                })
            
            return JsonResponse({
                'success': True,
                'folder_stats': folder_stats
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error scanning custom folder: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def process_custom_folder(request):
    """Process all documents in a custom folder"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            folder_path = data.get('folder_path')
            recursive = data.get('recursive', True)
            
            if not folder_path:
                return JsonResponse({'error': 'Folder path is required'}, status=400)
            
            from pathlib import Path
            folder_path = Path(folder_path)
            
            if not folder_path.exists():
                return JsonResponse({'error': 'Folder does not exist'}, status=400)
            
            if not folder_path.is_dir():
                return JsonResponse({'error': 'Path is not a directory'}, status=400)
            
            # Process all documents in the folder
            documents = document_processor.process_folder(folder_path, recursive=recursive)
            
            processed_count = 0
            failed_count = 0
            
            for doc in documents:
                try:
                    success = rag_service.embed_document(doc)
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.title}: {str(e)}")
                    failed_count += 1
            
            return JsonResponse({
                'success': True,
                'processed_count': processed_count,
                'failed_count': failed_count,
                'total_documents': len(documents)
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error processing custom folder: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def clear_all_documents(request):
    """Clear all documents from the system"""
    if request.method == 'POST':
        try:
            # Get all document IDs first
            document_ids = list(RAGDocument.objects.filter(is_active=True).values_list('id', flat=True))
            
            deleted_count = 0
            
            # Delete all documents and their embeddings
            for doc_id in document_ids:
                try:
                    # Delete embeddings first
                    rag_service.delete_document_embeddings(doc_id)
                    
                    # Delete the document
                    RAGDocument.objects.filter(id=doc_id).delete()
                    deleted_count += 1
                    
                except Exception as e:
                    logger.error(f"Error deleting document {doc_id}: {str(e)}")
            
            return JsonResponse({
                'success': True,
                'deleted_count': deleted_count
            })
            
        except Exception as e:
            logger.error(f"Error clearing all documents: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def scan_folder(request):
    """Scan folder and return statistics"""
    if request.method == 'GET':
        try:
            folder_stats = document_processor.scan_folder()
            return JsonResponse({
                'success': True,
                'folder_stats': folder_stats
            })
            
        except Exception as e:
            logger.error(f"Error scanning folder: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def process_all_unprocessed(request):
    """Process all unprocessed documents in the folder"""
    if request.method == 'POST':
        try:
            # Get unprocessed files
            folder_stats = document_processor.scan_folder()
            unprocessed_files = folder_stats.get('unprocessed_files', [])
            
            processed_count = 0
            failed_count = 0
            
            for file_info in unprocessed_files:
                try:
                    file_path = Path(file_info['full_path'])
                    doc = document_processor.process_file(file_path)
                    
                    if doc:
                        # Start embedding process
                        success = rag_service.embed_document(doc)
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {file_info['path']}: {str(e)}")
                    failed_count += 1
            
            if is_htmx_request(request):
                return HttpResponse(f"""
                    <div class="alert alert-success alert-dismissible fade show">
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                        Processed {processed_count} documents successfully, {failed_count} failed
                    </div>
                """)
            else:
                return JsonResponse({
                    'success': True,
                    'processed_count': processed_count,
                    'failed_count': failed_count,
                    'message': f'Processed {processed_count} documents successfully, {failed_count} failed'
                })
            
        except Exception as e:
            logger.error(f"Error batch processing documents: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def refresh_folder_scan(request):
    """Refresh folder scan and return updated statistics"""
    if request.method == 'POST':
        try:
            # Force refresh by re-scanning the folder
            folder_stats = document_processor.scan_folder()
            
            # Also get updated database stats
            documents = RAGDocument.objects.filter(is_active=True)
            total_documents = documents.count()
            total_chunks = RAGEmbedding.objects.count()
            completed_embeddings = documents.filter(embedding_status='completed').count()
            
            if is_htmx_request(request):
                # Return updated folder stats HTML fragment
                return render(request, 'chat_rag/partials/folder_stats.html', {
                    'folder_stats': folder_stats
                })
            else:
                return JsonResponse({
                    'success': True,
                    'folder_stats': folder_stats,
                    'db_stats': {
                        'total_documents': total_documents,
                        'total_chunks': total_chunks,
                        'completed_embeddings': completed_embeddings
                    }
                })
            
        except Exception as e:
            logger.error(f"Error refreshing folder scan: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
