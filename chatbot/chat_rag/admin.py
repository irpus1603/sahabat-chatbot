from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import RAGDocument, RAGQuery, ChatMessage, ChatRAGContext, RAGEmbedding


@admin.register(RAGDocument)
class RAGDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'embedding_status', 'chunk_count', 'file_size_display', 'created_at', 'actions_display')
    list_filter = ('embedding_status', 'is_active', 'created_at')
    search_fields = ('title', 'source')
    readonly_fields = ('created_at', 'updated_at', 'content_preview', 'metadata_display')
    list_per_page = 20
    ordering = ('-created_at',)
    
    fieldsets = (
        ('Document Information', {
            'fields': ('title', 'source', 'embedding_status', 'is_active')
        }),
        ('Content', {
            'fields': ('content_preview',),
            'classes': ('collapse',)
        }),
        ('Processing Details', {
            'fields': ('chunk_count', 'metadata_display'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def file_size_display(self, obj):
        """Display file size in human readable format"""
        try:
            size = obj.metadata.get('file_size', 0)
            if size == 0:
                return '-'
            
            # Convert bytes to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return '-'
    file_size_display.short_description = 'File Size'
    
    def content_preview(self, obj):
        """Show preview of document content"""
        if obj.content:
            preview = obj.content[:500] + ('...' if len(obj.content) > 500 else '')
            return format_html('<pre style="white-space: pre-wrap;">{}</pre>', preview)
        return '-'
    content_preview.short_description = 'Content Preview'
    
    def metadata_display(self, obj):
        """Display metadata as formatted JSON"""
        import json
        try:
            metadata = json.dumps(obj.metadata, indent=2)
            return format_html('<pre>{}</pre>', metadata)
        except:
            return str(obj.metadata)
    metadata_display.short_description = 'Metadata'
    
    def actions_display(self, obj):
        """Display action buttons"""
        actions = []
        
        if obj.embedding_status == 'failed' or obj.embedding_status == 'pending':
            actions.append(
                format_html(
                    '<a class="button" href="javascript:void(0)" onclick="reprocessDocument({})">Reprocess</a>',
                    obj.id
                )
            )
        
        
        # Link to view embeddings
        embeddings_url = reverse('admin:chat_rag_ragembedding_changelist') + f'?document__id={obj.id}'
        embeddings_count = obj.embeddings.count() if hasattr(obj, '_prefetched_objects_cache') else obj.chunk_count
        actions.append(
            format_html(
                '<a class="button" href="{}">View Embeddings ({})</a>',
                embeddings_url,
                embeddings_count
            )
        )
        
        return mark_safe(' | '.join(actions))
    actions_display.short_description = 'Actions'
    
    def get_queryset(self, request):
        """Optimize queries"""
        return super().get_queryset(request).select_related()



@admin.register(RAGEmbedding)
class RAGEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('id', 'document_title', 'chunk_index', 'content_preview', 'embedding_size', 'created_at')
    list_filter = ('document__embedding_status', 'created_at', 'document__title')
    search_fields = ('document__title', 'content', 'document__source')
    readonly_fields = ('created_at', 'content_display', 'metadata_display', 'embedding_info')
    list_per_page = 50
    ordering = ('document_id', 'chunk_index')
    
    fieldsets = (
        ('Embedding Information', {
            'fields': ('document', 'chunk_index')
        }),
        ('Content', {
            'fields': ('content_display',)
        }),
        ('Embedding Data', {
            'fields': ('embedding_info',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata_display',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def document_title(self, obj):
        """Show document title with link"""
        return format_html(
            '<a href="{}">{}</a>',
            reverse('admin:chat_rag_ragdocument_change', args=[obj.document.id]),
            obj.document.title[:50] + ('...' if len(obj.document.title) > 50 else '')
        )
    document_title.short_description = 'Document'
    document_title.admin_order_field = 'document__title'
    
    def content_preview(self, obj):
        """Show content preview"""
        preview = obj.content[:80] + ('...' if len(obj.content) > 80 else '')
        return format_html('<span title="{}">{}</span>', obj.content, preview)
    content_preview.short_description = 'Content Preview'
    
    def content_display(self, obj):
        """Display full content"""
        return format_html('<pre style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">{}</pre>', obj.content)
    content_display.short_description = 'Full Content'
    
    def embedding_size(self, obj):
        """Show embedding size in bytes"""
        if obj.embedding:
            size = len(obj.embedding)
            if size > 1024:
                return f"{size / 1024:.1f} KB"
            return f"{size} bytes"
        return '-'
    embedding_size.short_description = 'Embedding Size'
    
    def embedding_info(self, obj):
        """Display embedding information"""
        if obj.embedding:
            import numpy as np
            try:
                # Convert embedding back to numpy array to get shape info
                embedding_array = np.frombuffer(obj.embedding, dtype=np.float32)
                info = f"Dimensions: {embedding_array.shape[0]}\n"
                info += f"Size: {len(obj.embedding)} bytes\n"
                info += f"Sample values: {embedding_array[:5].tolist()}\n"
                info += f"Min: {embedding_array.min():.6f}\n"
                info += f"Max: {embedding_array.max():.6f}\n"
                info += f"Mean: {embedding_array.mean():.6f}"
                return format_html('<pre>{}</pre>', info)
            except Exception as e:
                return f"Error reading embedding: {e}"
        return '-'
    embedding_info.short_description = 'Embedding Details'
    
    def metadata_display(self, obj):
        """Display metadata as formatted JSON"""
        import json
        try:
            metadata = json.dumps(obj.metadata, indent=2)
            return format_html('<pre>{}</pre>', metadata)
        except:
            return str(obj.metadata)
    metadata_display.short_description = 'Metadata'
    
    def get_queryset(self, request):
        """Optimize queries"""
        return super().get_queryset(request).select_related('document')
    
    actions = ['test_similarity_search']
    
    def test_similarity_search(self, request, queryset):
        """Test similarity search with selected embeddings"""
        if queryset.count() > 1:
            self.message_user(request, "Please select only one embedding to test similarity search.", level='warning')
            return
        
        embedding_obj = queryset.first()
        if not embedding_obj:
            return
            
        # Import here to avoid circular imports
        from .chat_rag import rag_service
        
        try:
            # Use the content as query to find similar embeddings
            query = embedding_obj.content[:100]  # Use first 100 chars as query
            results = rag_service.search_similar(query, top_k=5)
            
            message = f"Similarity search results for '{query[:50]}...':\n"
            for i, result in enumerate(results[:3]):
                score = result.get('relevance_score', 0)
                content_preview = result.get('content', '')[:50]
                message += f"{i+1}. Score: {score:.3f} - {content_preview}...\n"
            
            self.message_user(request, message, level='info')
            
        except Exception as e:
            self.message_user(request, f"Error testing similarity search: {e}", level='error')
    
    test_similarity_search.short_description = "Test similarity search with selected embedding"


@admin.register(RAGQuery)
class RAGQueryAdmin(admin.ModelAdmin):
    list_display = ('query_preview', 'user', 'results_count', 'avg_distance', 'processing_time', 'created_at')
    list_filter = ('created_at', 'results_count')
    search_fields = ('query', 'user__username')
    readonly_fields = ('created_at', 'query_display')
    list_per_page = 50
    ordering = ('-created_at',)
    
    fieldsets = (
        ('Query Information', {
            'fields': ('user', 'query_display', 'results_count')
        }),
        ('Performance Metrics', {
            'fields': ('avg_distance', 'processing_time')
        }),
        ('Timestamp', {
            'fields': ('created_at',)
        }),
    )
    
    def query_preview(self, obj):
        """Show query preview"""
        preview = obj.query[:80] + ('...' if len(obj.query) > 80 else '')
        return format_html('<span title="{}">{}</span>', obj.query, preview)
    query_preview.short_description = 'Query'
    
    def query_display(self, obj):
        """Display full query"""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.query)
    query_display.short_description = 'Full Query'
    
    def get_queryset(self, request):
        """Optimize queries"""
        return super().get_queryset(request).select_related('user')


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('conversation_display', 'message_type', 'content_preview', 'timestamp')
    list_filter = ('message_type', 'timestamp')
    search_fields = ('content', 'conversation__id')
    readonly_fields = ('timestamp', 'content_display')
    list_per_page = 50
    ordering = ('-timestamp',)
    
    fieldsets = (
        ('Message Information', {
            'fields': ('conversation', 'message_type')
        }),
        ('Content', {
            'fields': ('content_display',)
        }),
        ('Timestamp', {
            'fields': ('timestamp',)
        }),
    )
    
    def content_preview(self, obj):
        """Show content preview"""
        preview = obj.content[:100] + ('...' if len(obj.content) > 100 else '')
        return format_html('<span title="{}">{}</span>', obj.content, preview)
    content_preview.short_description = 'Content Preview'
    
    def content_display(self, obj):
        """Display full content"""
        return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.content)
    content_display.short_description = 'Full Content'
    
    def conversation_display(self, obj):
        """Display conversation ID with link"""
        return format_html(
            '<a href="{}">{}</a>',
            reverse('admin:app_chat_change', args=[obj.conversation.id]),
            f'Chat {obj.conversation.id}'
        )
    conversation_display.short_description = 'Conversation'
    conversation_display.admin_order_field = 'conversation__id'


@admin.register(ChatRAGContext)
class ChatRAGContextAdmin(admin.ModelAdmin):
    list_display = ('chat_message_display', 'document_title', 'relevance_score', 'used_in_response')
    list_filter = ('relevance_score', 'used_in_response')
    search_fields = ('chat_message__conversation__id', 'embedding__document__title')
    readonly_fields = ('content_display',)
    list_per_page = 50
    ordering = ('-relevance_score',)
    
    fieldsets = (
        ('Context Information', {
            'fields': ('chat_message', 'embedding', 'relevance_score', 'used_in_response')
        }),
        ('Content', {
            'fields': ('content_display',)
        }),
    )
    
    def document_title(self, obj):
        """Show document title with link"""
        if obj.embedding and obj.embedding.document:
            return format_html(
                '<a href="{}">{}</a>',
                reverse('admin:chat_rag_ragdocument_change', args=[obj.embedding.document.id]),
                obj.embedding.document.title[:60] + ('...' if len(obj.embedding.document.title) > 60 else '')
            )
        return '-'
    document_title.short_description = 'Document'
    document_title.admin_order_field = 'embedding__document__title'
    
    def chat_message_display(self, obj):
        """Display chat message with link"""
        if obj.chat_message:
            return format_html(
                '<a href="{}">{}</a>',
                reverse('admin:chat_rag_chatmessage_change', args=[obj.chat_message.id]),
                f'Message {obj.chat_message.id}'
            )
        return '-'
    chat_message_display.short_description = 'Chat Message'
    chat_message_display.admin_order_field = 'chat_message__id'
    
    def content_display(self, obj):
        """Display retrieved content"""
        if obj.embedding:
            return format_html('<pre style="white-space: pre-wrap;">{}</pre>', obj.embedding.content)
        return '-'
    content_display.short_description = 'Retrieved Content'
    
    def get_queryset(self, request):
        """Optimize queries"""
        return super().get_queryset(request).select_related('embedding__document', 'chat_message')


# Add custom admin site configuration
admin.site.site_header = 'RAG System Administration'
admin.site.site_title = 'RAG Admin'
admin.site.index_title = 'RAG System Management'

# Add custom CSS and JavaScript for better UX
class RAGAdminMixin:
    """Mixin to add custom styling to RAG admin pages"""
    
    class Media:
        css = {
            'all': ('admin/css/rag_admin.css',)
        }
        js = ('admin/js/rag_admin.js',)


# Apply mixin to all RAG admin classes
RAGDocumentAdmin.__bases__ = (RAGAdminMixin,) + RAGDocumentAdmin.__bases__
RAGEmbeddingAdmin.__bases__ = (RAGAdminMixin,) + RAGEmbeddingAdmin.__bases__
RAGQueryAdmin.__bases__ = (RAGAdminMixin,) + RAGQueryAdmin.__bases__
ChatMessageAdmin.__bases__ = (RAGAdminMixin,) + ChatMessageAdmin.__bases__
ChatRAGContextAdmin.__bases__ = (RAGAdminMixin,) + ChatRAGContextAdmin.__bases__
