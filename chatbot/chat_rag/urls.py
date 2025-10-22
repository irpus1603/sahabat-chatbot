from django.urls import path
from . import views

urlpatterns = [
    # Document management
    path('documents/', views.document_upload, name='document_upload'),
    path('documents/process-folder/', views.process_folder, name='process_folder'),
    path('documents/<int:doc_id>/reprocess/', views.reprocess_document, name='reprocess_document'),
    path('documents/<int:doc_id>/delete/', views.delete_document, name='delete_document'),
    
    # Folder scanning and batch processing
    path('documents/scan-folder/', views.scan_folder, name='scan_folder'),
    path('documents/scan-custom/', views.scan_custom_folder, name='scan_custom_folder'),
    path('documents/process-custom/', views.process_custom_folder, name='process_custom_folder'),
    path('documents/process-all/', views.process_all_unprocessed, name='process_all_unprocessed'),
    path('documents/clear-all/', views.clear_all_documents, name='clear_all_documents'),
    path('documents/refresh-scan/', views.refresh_folder_scan, name='refresh_folder_scan'),
    
    # Document search
    path('search/', views.document_search, name='document_search'),
    
    # RAG chat API
    path('chat/api/', views.rag_chat_api, name='rag_chat_api'),
    path('chat/stream/', views.rag_chat_stream_api, name='rag_chat_stream_api'),

    # Analytics
    path('analytics/', views.rag_analytics, name='rag_analytics'),
]