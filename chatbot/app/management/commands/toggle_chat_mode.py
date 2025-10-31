from django.core.management.base import BaseCommand
from django.conf import settings
import os


class Command(BaseCommand):
    help = 'Toggle between fast chat and LangGraph chat modes'

    def add_arguments(self, parser):
        parser.add_argument('--mode', type=str, choices=['fast', 'langgraph'], 
                          help='Chat mode to use: fast or langgraph')
        parser.add_argument('--status', action='store_true', 
                          help='Show current chat mode status')

    def handle(self, *args, **options):
        if options.get('status'):
            self.show_status()
        elif options.get('mode'):
            self.set_mode(options['mode'])
        else:
            self.stdout.write(self.style.ERROR('Please specify --mode or --status'))

    def show_status(self):
        """Show current chat implementation status."""
        self.stdout.write(self.style.SUCCESS('=== Chat Implementation Status ==='))
        
        try:
            # Test fast chat
            from app.fast_chat import fast_health_check
            fast_health = fast_health_check()
            self.stdout.write(f"Fast Chat: {fast_health.get('status', 'unknown')}")
            
            # Test LangGraph chat
            from app.chat import health_check_enhanced
            langgraph_health = health_check_enhanced()
            self.stdout.write(f"LangGraph Chat: {langgraph_health.get('overall_status', 'unknown')}")
            
            # Check which is currently being used in views
            with open('app/views.py', 'r') as f:
                content = f.read()
                if 'create_fast_llm_instance()' in content:
                    current_mode = 'fast'
                elif 'create_enhanced_llm_instance(use_langgraph=True)' in content:
                    current_mode = 'langgraph'
                else:
                    current_mode = 'unknown'
            
            self.stdout.write(f"Current Mode: {current_mode}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error checking status: {e}'))

    def set_mode(self, mode):
        """Set chat mode by updating views.py."""
        self.stdout.write(f'Setting chat mode to: {mode}')
        
        try:
            views_path = 'app/views.py'
            
            # Read current views.py
            with open(views_path, 'r') as f:
                content = f.read()
            
            if mode == 'fast':
                # Replace LangGraph calls with fast calls
                content = content.replace(
                    'create_enhanced_llm_instance(use_langgraph=True)',
                    'create_fast_llm_instance()'
                )
                content = content.replace(
                    'LLM response generated for user',
                    'Fast LLM response generated for user'
                )
                
            elif mode == 'langgraph':
                # Replace fast calls with LangGraph calls  
                content = content.replace(
                    'create_fast_llm_instance()',
                    'create_enhanced_llm_instance(use_langgraph=True)'
                )
                content = content.replace(
                    'Fast LLM response generated for user',
                    'LLM response generated for user'
                )
            
            # Write updated content
            with open(views_path, 'w') as f:
                f.write(content)
            
            self.stdout.write(self.style.SUCCESS(f'Successfully switched to {mode} mode'))
            self.stdout.write('Please restart your Django server for changes to take effect.')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error setting mode: {e}'))