from django.urls import path
from . import views
from .test_chat_view import test_chat_api

urlpatterns = [
    path('', views.new_chatbot_view, name='home'),  # Make new chatbot the main home page
    path('chat/', views.chat, name='chat'),
    path('chat/<int:chat_id>/', views.chat, name='chat_detail'),
    path('<int:chat_id>/', views.new_chatbot_view, name='chat_with_id'),  # Load specific chat at root level
    path('api/chat/', views.chat_api, name='chat_api'),  # AJAX endpoint for real-time chat
    path('api/chat/stream/', views.chat_stream_api, name='chat_stream_api'),  # Streaming endpoint for real-time chat
    path('api/test-chat/', test_chat_api, name='test_chat_api'),  # Test endpoint for debugging
    path('api/test-standard-chat/', views.test_standard_chat, name='test_standard_chat'),  # Test standard LangChain chat
    path('chat/<int:chat_id>/clear-memory/', views.clear_chat_memory, name='clear_chat_memory'),
    path('chat/<int:chat_id>/delete/', views.delete_chat, name='delete_chat'),
    path('clear-all-history/', views.clear_all_chat_history, name='clear_all_chat_history'),
    path('new-chat/', views.new_chat, name='new_chat'),
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings_view, name='settings'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('toggle-rag-mode/', views.toggle_rag_mode, name='toggle_rag_mode'),
    path('set-prompt/', views.set_prompt, name='set_prompt'),
    path('toggle-sidebar-state/', views.toggle_sidebar_state, name='toggle_sidebar_state'),
    path('old-home/', views.home, name='old_home'),  # Keep old home for reference
    path('new-chatbot/', views.new_chatbot_view, name='new_chatbot'),  # Alternative URL
    path('new-chatbot/<int:chat_id>/', views.new_chatbot_view, name='new_chatbot_with_id'),  # Alternative URL

    #LLM Agent Model
    path('chat/case-mgt/', views.chat_rag_case, name='chat_rag_case'),
]

 