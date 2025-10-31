"""
Test chat view for debugging timeout issues.
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
from .minimal_chat import get_minimal_chat_manager

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def test_chat_api(request):
    """
    Test chat API endpoint with minimal implementation.
    """
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        
        if not message:
            return JsonResponse({'error': 'Message required'}, status=400)
        
        logger.info(f"Received test message: {message}")
        
        # Use minimal chat manager
        chat_manager = get_minimal_chat_manager()
        
        logger.info("About to generate response...")
        response = chat_manager.generate_response(message)
        logger.info(f"Generated response: {response[:100]}...")
        
        return JsonResponse({
            'success': True,
            'message': message,
            'response': response,
            'implementation': 'minimal'
        })
        
    except Exception as e:
        logger.error(f"Test chat API error: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)