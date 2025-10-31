from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.conf import settings
from django.template.loader import render_to_string
import logging
import json
from .models import Chat, Message
from .chat import SahabatLLM, ENV_CONFIG, GLOBAL_MEMORY_MANAGER, create_llm_instance
from chat_rag.rag_chat import rag_chat_service

# Get logger for this module
logger = logging.getLogger(__name__)

def is_htmx_request(request):
    """Check if the request is from HTMX"""
    return request.headers.get('HX-Request') == 'true'


def home(request):
    if request.user.is_authenticated:
        chat_history = Chat.objects.filter(user=request.user)[:10]
        return render(request, 'app/index.html', {
            'chat_history': chat_history,
            'messages': []
        })
    return render(request, 'app/index.html', {'chat_history': [], 'messages': []})

@login_required
def chat(request, chat_id=None):
    if request.method == 'POST':
        message_content = request.POST.get('message', '').strip()
        if message_content:
            if chat_id:
                chat_obj = get_object_or_404(Chat, id=chat_id, user=request.user)
            else:
                chat_obj = Chat.objects.create(user=request.user, title=message_content[:50])
            
            Message.objects.create(
                chat=chat_obj,
                sender='user',
                content=message_content
            )
            
            # Generate AI response using RAG or standard LLM based on mode
            try:
                # Check if RAG mode is enabled
                rag_mode_enabled = request.session.get('rag_mode_enabled', False)
                
                if rag_mode_enabled:
                    # Use RAG-enhanced chat
                    assistant_response = rag_chat_service.generate_rag_response(
                        message_content, 
                        chat_id=str(chat_obj.id), 
                        user_id=request.user.id
                    )
                    logger.info(f"RAG response generated for user {request.user.username}")
                else:
                    # Use standard LLM implementation with conversation memory
                    llm = create_llm_instance()
                    assistant_response = llm.generate_response(message_content, chat_id=str(chat_obj.id))
                    logger.info(f"Standard LLM response generated for user {request.user.username}")
                
            except Exception as e:
                # Log the error and provide fallback response
                logger.error(f"Chat request failed: {str(e)}")
                assistant_response = "I'm sorry, I'm experiencing technical difficulties right now. Please try again later."
            
            # Save assistant response to database
            Message.objects.create(
                chat=chat_obj,
                sender='assistant',
                content=assistant_response
            )
            
            return redirect('chat_detail', chat_id=chat_obj.id)
    
    if chat_id:
        chat_obj = get_object_or_404(Chat, id=chat_id, user=request.user)
        messages = chat_obj.messages.all()
    else:
        messages = []
    
    chat_history = Chat.objects.filter(user=request.user)[:10]
    return render(request, 'app/index.html', {
        'chat_history': chat_history,
        'messages': messages
    })

@login_required
def new_chat(request):
    return redirect('home')  # Now redirects to new chatbot page

# new rag chat with case management agent
@login_required
def chat_rag_case(request):
    return render(request, 'app/chat_rag_case.html')

@login_required
def profile(request):
    return render(request, 'app/profile.html')

@login_required
def settings_view(request):
    return render(request, 'app/settings.html')

def new_chatbot_view(request, chat_id=None):
    """
    View for the new professional chatbot design page
    """
    messages = []
    current_chat = None
    
    if request.user.is_authenticated:
        chat_history = Chat.objects.filter(user=request.user).order_by('-created_at')[:10]
        
        # If chat_id is provided, load that specific chat
        if chat_id:
            try:
                current_chat = Chat.objects.get(id=chat_id, user=request.user)
                messages = current_chat.messages.all()
            except Chat.DoesNotExist:
                # Chat not found or doesn't belong to user, redirect to new chat
                return redirect('home')
        
        return render(request, 'app/new_chatbot.html', {
            'chat_history': chat_history,
            'messages': messages,
            'current_chat': current_chat,
        })
    return render(request, 'app/new_chatbot.html', {'chat_history': [], 'messages': []})

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        messages.error(request, 'Invalid credentials')
    return render(request, 'app/login.html')

def logout_view(request):
    logout(request)
    return redirect('home')

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'app/register.html', {'form': form})

@login_required
def chat_api(request):
    """
    API endpoint for sending chat messages and receiving responses.
    Supports both HTMX form submissions and JSON AJAX requests.
    """
    if request.method == 'POST':
        try:
            # Handle both HTMX form data and JSON requests
            if is_htmx_request(request):
                message_content = request.POST.get('message', '').strip()
                chat_id = request.POST.get('chat_id')
            else:
                data = json.loads(request.body)
                message_content = data.get('message', '').strip()
                chat_id = data.get('chat_id')
            
            if not message_content:
                return JsonResponse({'error': 'Message content is required'}, status=400)
            
            # Get or create chat
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
            
            # Generate AI response using RAG or standard LLM based on mode
            try:
                # Check if RAG mode is enabled
                rag_mode_enabled = request.session.get('rag_mode_enabled', False)
                
                if rag_mode_enabled:
                    # Use RAG-enhanced chat
                    assistant_response = rag_chat_service.generate_rag_response(
                        message_content, 
                        chat_id=str(chat_obj.id), 
                        user_id=request.user.id
                    )
                    logger.info(f"RAG response generated for user {request.user.username} in chat {chat_obj.id}")
                else:
                    # Use standard LLM implementation with conversation memory
                    llm = create_llm_instance()
                    assistant_response = llm.generate_response(message_content, chat_id=str(chat_obj.id))
                    logger.info(f"Standard LLM response generated for user {request.user.username} in chat {chat_obj.id}")
                
            except Exception as e:
                # Log the error and provide fallback response
                logger.error(f"Chat request failed for user {request.user.username}: {str(e)}")
                assistant_response = "I'm sorry, I'm experiencing technical difficulties right now. Please try again later."
            
            # Save assistant response to database
            assistant_message = Message.objects.create(
                chat=chat_obj,
                sender='assistant',
                content=assistant_response
            )
            
            # Return appropriate response based on request type
            if is_htmx_request(request):
                # Return HTML fragments for HTMX
                user_html = render_to_string('app/partials/message.html', {
                    'message': user_message,
                    'user': request.user
                })
                assistant_html = render_to_string('app/partials/message.html', {
                    'message': assistant_message,
                    'user': request.user
                })
                return HttpResponse(user_html + assistant_html)
            else:
                # Return JSON response for AJAX
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
            logger.error(f"Chat API error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def clear_chat_memory(request, chat_id):
    """
    Clear conversation memory for a specific chat.
    """
    try:
        # Verify user owns this chat
        chat_obj = get_object_or_404(Chat, id=chat_id, user=request.user)
        
        # Clear memory using standard LangChain memory manager
        try:
            GLOBAL_MEMORY_MANAGER.clear_memory(str(chat_id))
        except Exception as e:
            logger.warning(f"Failed to clear memory for chat {chat_id}: {e}")
        
        logger.info(f"Cleared conversation memory for chat {chat_id}")
        
        if request.method == 'POST':
            return JsonResponse({'success': True, 'message': 'Chat memory cleared'})
        else:
            messages.success(request, 'Chat memory cleared successfully')
            return redirect('chat_detail', chat_id=chat_id)
            
    except Exception as e:
        logger.error(f"Error clearing chat memory: {str(e)}")
        if request.method == 'POST':
            return JsonResponse({'error': 'Failed to clear memory'}, status=500)
        else:
            messages.error(request, 'Failed to clear chat memory')
            return redirect('home')

@login_required
def delete_chat(request, chat_id):
    """
    Delete a specific chat and all its messages.
    """
    if request.method == 'POST':
        try:
            # Verify user owns this chat
            chat_obj = get_object_or_404(Chat, id=chat_id, user=request.user)
            
            # Clear memory for this chat using standard LangChain memory manager
            try:
                GLOBAL_MEMORY_MANAGER.clear_memory(str(chat_id))
            except Exception as e:
                logger.warning(f"Failed to clear memory for chat {chat_id}: {e}")
            
            # Delete the chat (will cascade delete messages)
            chat_obj.delete()
            
            logger.info(f"Chat {chat_id} deleted by user {request.user.username}")
            
            if is_htmx_request(request):
                # Return empty response to remove the element
                return HttpResponse("")
            else:
                return JsonResponse({'success': True, 'message': 'Chat deleted successfully'})
            
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {str(e)}")
            return JsonResponse({'error': 'Failed to delete chat'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def clear_all_chat_history(request):
    """
    Clear all chat history for the current user.
    """
    if request.method == 'POST':
        try:
            # Get all chats for the current user
            user_chats = Chat.objects.filter(user=request.user)
            
            # Clear memory for all chats using standard LangChain memory manager
            for chat in user_chats:
                try:
                    GLOBAL_MEMORY_MANAGER.clear_memory(str(chat.id))
                except Exception as e:
                    logger.warning(f"Failed to clear memory for chat {chat.id}: {e}")
            
            # Delete all chats (will cascade delete messages)
            chat_count = user_chats.count()
            user_chats.delete()
            
            logger.info(f"All chat history cleared for user {request.user.username} ({chat_count} chats)")
            
            return JsonResponse({
                'success': True, 
                'message': f'All chat history cleared ({chat_count} conversations deleted)'
            })
            
        except Exception as e:
            logger.error(f"Error clearing all chat history for user {request.user.username}: {str(e)}")
            return JsonResponse({'error': 'Failed to clear chat history'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def test_standard_chat(request):
    """
    Test endpoint for standard LangChain chat.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', 'Hello, this is a test')
            
            # Test standard LLM with LangChain memory
            llm = create_llm_instance()
            
            import time
            start_time = time.time()
            
            response = llm.generate_response(
                message, 
                chat_id="test_standard_123"
            )
            
            elapsed_time = time.time() - start_time
            
            # Get memory stats
            stats = llm.get_memory_stats()
            
            return JsonResponse({
                'success': True,
                'response': response,
                'elapsed_time': round(elapsed_time, 2),
                'memory_type': 'LangChain ConversationBufferMemory',
                'stats': stats
            })
            
        except Exception as e:
            logger.error(f"Test standard chat error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def toggle_rag_mode(request):
    """Toggle RAG mode via HTMX"""
    if request.method == 'POST':
        # Toggle the RAG mode state in session
        current_state = request.session.get('rag_mode_enabled', False)
        new_state = not current_state
        request.session['rag_mode_enabled'] = new_state
        
        if is_htmx_request(request):
            # Return updated button HTML with proper context
            from django.middleware.csrf import get_token
            csrf_token = get_token(request)
            
            button_class = "rag-mode-on" if new_state else "rag-mode-off"
            icon_class = "toggle-on" if new_state else "toggle-off"
            status_text = "RAG Mode: ON" if new_state else "RAG Mode: OFF"
            
            button_html = f'''
                <button class="btn rag-toggle-btn btn-sm d-flex align-items-center justify-content-center {button_class}" 
                        hx-post="/toggle-rag-mode/" 
                        hx-target="#ragToggleContainer"
                        hx-swap="innerHTML"
                        hx-headers='{{"X-CSRFToken": "{csrf_token}"}}'>
                    <i class="bi bi-{icon_class} me-2"></i>
                    <span>{status_text}</span>
                </button>
            '''
            return HttpResponse(button_html)
        else:
            return JsonResponse({'success': True, 'rag_enabled': new_state})
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def set_prompt(request):
    """Set prompt in textarea via HTMX"""
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        if is_htmx_request(request):
            # Return the textarea with the prompt set
            return HttpResponse(f"""
                <textarea 
                    class="form-control chat-input border rounded-4 pe-5" 
                    name="message" 
                    id="messageInput"
                    placeholder="Message to Sahabat Bot..." 
                    rows="2"
                    style="resize: vertical; min-height: 72px; max-height: 200px; padding: 0.75rem 3rem 0.75rem 1rem;"
                    hx-trigger="keydown[key=='Enter' && !shiftKey] from:body"
                    hx-post="/api/chat/"
                    hx-target="#chatMessages"
                    hx-swap="beforeend"
                    hx-include="closest form"
                    required autofocus>{prompt}</textarea>
            """)
        else:
            return JsonResponse({'success': True, 'prompt': prompt})
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def toggle_sidebar_state(request):
    """Toggle sidebar state and store in session"""
    if request.method == 'POST':
        # Toggle the state
        current_state = request.session.get('sidebar_collapsed', False)
        request.session['sidebar_collapsed'] = not current_state
        
        if is_htmx_request(request):
            return HttpResponse("")  # Empty response for HTMX
        else:
            return JsonResponse({'success': True, 'collapsed': not current_state})
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def chat_stream_api(request):
    """
    Streaming API endpoint for real-time chat responses.
    Uses Server-Sent Events (SSE) for streaming LLM responses.
    """
    if request.method == 'POST':
        try:
            # Handle form data (from FormData in JavaScript)
            message_content = request.POST.get('message', '').strip()
            chat_id = request.POST.get('chat_id', '').strip() or None
            
            if not message_content:
                return JsonResponse({'error': 'Message content is required'}, status=400)
            
            # Get or create chat
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
            
            def stream_response():
                """Generator function for streaming response"""
                try:
                    # Send initial response with user message
                    yield f"data: {json.dumps({'type': 'user_message', 'content': message_content, 'chat_id': chat_obj.id})}\n\n"
                    
                    # Start assistant response
                    yield f"data: {json.dumps({'type': 'assistant_start'})}\n\n"
                    
                    # Generate AI response using RAG or standard LLM based on mode
                    rag_mode_enabled = request.session.get('rag_mode_enabled', False)
                    
                    if rag_mode_enabled:
                        # Use RAG-enhanced chat with streaming
                        assistant_response = ""

                        for chunk in rag_chat_service.generate_rag_response_streaming(
                            query=message_content,
                            chat_id=str(chat_obj.id),
                            user_id=request.user.id
                        ):
                            assistant_response += chunk
                            # Send each chunk to the client
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                        logger.info(f"Streaming RAG response generated for user {request.user.username} in chat {chat_obj.id}")
                    else:
                        # Use standard LLM implementation with streaming
                        llm = create_llm_instance()
                        assistant_response = ""
                        
                        for chunk in llm.generate_streaming_response(message_content, chat_id=str(chat_obj.id)):
                            assistant_response += chunk
                            # Send each chunk to the client
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                        
                        logger.info(f"Streaming LLM response generated for user {request.user.username} in chat {chat_obj.id}")
                    
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
                    logger.error(f"Streaming chat request failed for user {request.user.username}: {str(e)}")
                    error_response = "I'm sorry, I'm experiencing technical difficulties right now. Please try again later."
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
            response = StreamingHttpResponse(
                stream_response(),
                content_type='text/event-stream'
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering

            return response

        except Exception as e:
            logger.error(f"Chat streaming API error: {str(e)}")
            return JsonResponse({'error': 'Internal server error'}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)