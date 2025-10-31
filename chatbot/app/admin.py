from django.contrib import admin
from .models import Chat, Message, LangGraphCheckpoint, LangGraphConversationState, LangGraphMemoryStats

@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ['user','title','created_at']
    search_fields = ['user', 'title']
    list_filter = ['user']


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ['chat','sender', 'content']


@admin.register(LangGraphCheckpoint)
class LangGraphCheckpointAdmin(admin.ModelAdmin):
    list_display = ['thread_id', 'checkpoint_id', 'created_at', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['thread_id', 'checkpoint_id']
    readonly_fields = ['created_at', 'updated_at']
    
    def has_add_permission(self, request):
        # Prevent manual creation of checkpoints
        return False


@admin.register(LangGraphConversationState)
class LangGraphConversationStateAdmin(admin.ModelAdmin):
    list_display = ['chat', 'thread_id', 'message_count', 'is_active', 'last_activity']
    list_filter = ['is_active', 'last_activity', 'created_at']
    search_fields = ['thread_id', 'chat__title', 'chat__user__username']
    readonly_fields = ['created_at', 'updated_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('chat', 'chat__user')


@admin.register(LangGraphMemoryStats)
class LangGraphMemoryStatsAdmin(admin.ModelAdmin):
    list_display = ['date', 'total_conversations', 'total_checkpoints', 'active_conversations', 'avg_response_time_ms']
    list_filter = ['date']
    readonly_fields = ['created_at']
    
    def has_add_permission(self, request):
        # Prevent manual creation of stats
        return False
