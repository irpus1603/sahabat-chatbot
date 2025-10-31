from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from app.models import Chat, Message
from app.chat import create_enhanced_llm_instance, health_check_enhanced


class Command(BaseCommand):
    help = 'Test LangGraph integration'

    def add_arguments(self, parser):
        parser.add_argument('--test-chat', action='store_true', help='Test chat functionality')
        parser.add_argument('--health-check', action='store_true', help='Run health check')
        parser.add_argument('--create-user', action='store_true', help='Create test user')

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Testing LangGraph Integration'))
        
        if options.get('health_check'):
            self.run_health_check()
            
        if options.get('create_user'):
            self.create_test_user()
            
        if options.get('test_chat'):
            self.test_chat_functionality()

    def run_health_check(self):
        self.stdout.write('Running health check...')
        health = health_check_enhanced()
        
        self.stdout.write(f"Overall Status: {health.get('overall_status', 'unknown')}")
        self.stdout.write(f"Standard LLM: {health.get('standard_llm', {}).get('status', 'unknown')}")
        self.stdout.write(f"LangGraph LLM: {health.get('langgraph_llm', {}).get('status', 'unknown')}")

    def create_test_user(self):
        self.stdout.write('Creating test user...')
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        if created:
            user.set_password('testpass123')
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Created user: {user.username}'))
        else:
            self.stdout.write(f'User already exists: {user.username}')

    def test_chat_functionality(self):
        self.stdout.write('Testing chat functionality...')
        
        # Get or create test user
        user, _ = User.objects.get_or_create(
            username='testuser',
            defaults={'email': 'test@example.com'}
        )
        
        # Create test chat
        chat = Chat.objects.create(
            user=user,
            title='Test LangGraph Chat'
        )
        
        # Test LangGraph LLM
        try:
            llm = create_enhanced_llm_instance(use_langgraph=True)
            
            # Test conversation
            test_message = "Hello! This is a test message for LangGraph integration."
            response = llm.generate_response(test_message, chat_id=str(chat.id))
            
            self.stdout.write(f"Test message: {test_message}")
            self.stdout.write(f"LangGraph response: {response}")
            
            # Save to database
            Message.objects.create(
                chat=chat,
                sender='user',
                content=test_message
            )
            
            Message.objects.create(
                chat=chat,
                sender='assistant',
                content=response
            )
            
            # Test memory stats
            stats = llm.get_memory_stats()
            self.stdout.write(f"Memory stats: {stats}")
            
            self.stdout.write(self.style.SUCCESS('LangGraph integration test completed successfully!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Test failed: {str(e)}'))
            raise