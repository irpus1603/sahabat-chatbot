"""
Django management command to process documents for RAG system.
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from pathlib import Path
import logging

from chat_rag.document_processor import document_processor
from chat_rag.chat_rag import rag_service
from chat_rag.models import RAGDocument

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Process documents for RAG system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--folder',
            type=str,
            help='Path to folder containing documents (defaults to configured folder)',
        )
        parser.add_argument(
            '--recursive',
            action='store_true',
            help='Process documents recursively in subdirectories',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reprocessing of already processed documents',
        )
        parser.add_argument(
            '--scan-only',
            action='store_true',
            help='Only scan and report statistics, do not process',
        )
        parser.add_argument(
            '--unprocessed-only',
            action='store_true',
            help='Only process unprocessed documents',
        )

    def handle(self, *args, **options):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        # Get folder path
        folder_path = None
        if options['folder']:
            folder_path = Path(options['folder'])
            if not folder_path.exists():
                raise CommandError(f"Folder {folder_path} does not exist")
        
        # Scan folder first
        self.stdout.write("Scanning folder for documents...")
        folder_stats = document_processor.scan_folder(folder_path)
        
        self.stdout.write(f"📁 Folder scan results:")
        self.stdout.write(f"  Total files: {folder_stats['total_files']}")
        self.stdout.write(f"  Supported files: {folder_stats['supported_files']}")
        self.stdout.write(f"  Unprocessed files: {len(folder_stats['unprocessed_files'])}")
        self.stdout.write(f"  Subdirectories: {len(folder_stats['folder_structure'])}")
        
        # Show file types
        if folder_stats['by_extension']:
            self.stdout.write(f"  File types:")
            for ext, count in folder_stats['by_extension'].items():
                self.stdout.write(f"    {ext}: {count}")
        
        # Show folder structure
        if folder_stats['folder_structure']:
            self.stdout.write(f"  Folder structure:")
            for folder in folder_stats['folder_structure']:
                self.stdout.write(f"    📁 {folder}")
        
        # If scan-only, exit here
        if options['scan_only']:
            return
        
        # Process documents
        if options['unprocessed_only']:
            self.stdout.write("\\n🔄 Processing unprocessed documents...")
            documents_to_process = []
            
            for file_info in folder_stats['unprocessed_files']:
                try:
                    file_path = Path(file_info['full_path'])
                    doc = document_processor.process_file(file_path)
                    if doc:
                        documents_to_process.append(doc)
                        self.stdout.write(f"  ✓ Processed: {file_info['path']}")
                    else:
                        self.stdout.write(f"  ✗ Failed to process: {file_info['path']}")
                except Exception as e:
                    self.stdout.write(f"  ✗ Error processing {file_info['path']}: {str(e)}")
        else:
            # Process all documents in folder
            self.stdout.write("\\n🔄 Processing all documents in folder...")
            documents_to_process = document_processor.process_folder(
                folder_path, 
                recursive=options['recursive']
            )
        
        # Generate embeddings
        if documents_to_process:
            self.stdout.write(f"\\n🧠 Generating embeddings for {len(documents_to_process)} documents...")
            
            success_count = 0
            failed_count = 0
            
            for doc in documents_to_process:
                try:
                    # Skip if already has embeddings and not forced
                    if doc.embedding_status == 'completed' and not options['force']:
                        self.stdout.write(f"  ⏩ Skipping {doc.title} (already processed)")
                        continue
                    
                    success = rag_service.embed_document(doc)
                    if success:
                        success_count += 1
                        self.stdout.write(f"  ✓ Embedded: {doc.title}")
                    else:
                        failed_count += 1
                        self.stdout.write(f"  ✗ Failed to embed: {doc.title}")
                        
                except Exception as e:
                    failed_count += 1
                    self.stdout.write(f"  ✗ Error embedding {doc.title}: {str(e)}")
            
            self.stdout.write(f"\\n📊 Processing complete:")
            self.stdout.write(f"  ✅ Successfully processed: {success_count}")
            self.stdout.write(f"  ❌ Failed: {failed_count}")
        else:
            self.stdout.write("\\n📝 No documents to process.")
        
        # Final statistics
        total_docs = RAGDocument.objects.filter(is_active=True).count()
        completed_docs = RAGDocument.objects.filter(
            is_active=True, 
            embedding_status='completed'
        ).count()
        
        self.stdout.write(f"\\n📈 System statistics:")
        self.stdout.write(f"  Total documents in system: {total_docs}")
        self.stdout.write(f"  Documents with embeddings: {completed_docs}")
        self.stdout.write(f"  Processing completion rate: {(completed_docs/total_docs*100):.1f}%" if total_docs > 0 else "  No documents in system")
        
        self.stdout.write("\\n🎉 Document processing completed!")