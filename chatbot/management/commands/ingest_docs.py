import os
from django.core.management.base import BaseCommand
from chatbot.rag.document_processor import process_document
from chatbot.rag.vector_store import add_chunks, get_stats, delete_collection
from chatbot.models import IngestedDocument


class Command(BaseCommand):
    help = 'Ingest Documents into the RAG vector store'

    def add_arguments(self, parser):
        """
        Define what arguments this command accepts.
        --file -> path the documents ingest
        --list -> show all ingested documents
        --clear -> wipe all vector
        """

        parser.add_argument(
            '--file',
            type=str,
            help='Path to documents to ingest (.pdf or .txt)'
        )

        parser.add_argument(
            '--list',
            action='store_true', # flag no view needed just presence
            help='List all ingested documents'
        )

        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear all vector from the store'
        )


    def handle(self, *args, **options):
        """
        Main logic - runs when command is excuted
        Checks when action was requested and run it 
        """

        # Action 1 - List ingested documents
        if options['list']:
            self.list_documents()

        
        # Action 2 - Clear all vectors 
        elif options['clear']:
            self.clear_documents()


        # Actions 3 - Ingested documents
        elif options['file']:
            self.ingest_document(options['file'])


        # No action specified
        else:
            self.stdout.write(
                self.style.WARNING(
                    "Please provide a action:\n"
                    "--file <path>  Ingested a documents\n"
                    "--list         List ingested documents\n"
                    "--clear        Clear all vectors"
                )
            )

    
    def ingest_document(self, filepath: str):
        """
        Full ingestion pipeline
        1 - check file exists
        2 - check not already ingestion
        3 - process into chunks
        4 - store into chromaDB
        5 - save record to DB
        """

        # step 1  - Check file exists
        if not os.path.exists(filepath):
            self.stdout.write(
                self.style.ERROR(f'File not found - {filepath}')
            )
            return 
        filename = os.path.basename(filepath)

        # step 2 check if already ingested
        # prevents Duplicate vector in ChromaDB

        if IngestedDocument.objects.filter(filename=filename).exists():
            self.stdout.write(
                self.style.WARNING(
                    f"'{filename}' already ingested"
                    f"Use --clear to re-ingest"
                )
            )
            return
        
        # Process documensts into chunks
        self.stdout.write(f"Processing: {filepath}")
        try:
            chunks = process_document(filepath)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to process: {e}"))
            return
        

        # step 4 store chunks in chromaDB
        self.stdout.write(f"Storing {len(chunks)} chunks in vector store")
        try:
            add_chunks(chunks)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to store: {e}'))
            return
        
        # step 5 Save record to Django  DB
        # so we know this documents has been ingested
        IngestedDocument.objects.create(
            filename=filename,
            filepath=filepath,
            chunk_count=len(chunks),
        )

        # show current stats
        stats = get_stats()
        self.stdout.write(
            f"Vector store now has {stats['total_chunks']} total chunks"
                
        )


    def list_documents(self):
        """show all documents that have been injested"""
        docs =  IngestedDocument.objects.all().order_by('-ingested_at')

        if not docs.exists():
            self.stdout.write(self.style.WARNING("No documents ingestes yet."))
            return
        
        self.stdout.write('\n Ingested Documents')
        self.stdout.write("-" * 50)

        for doc in docs:
            self.stdout.write(f"{doc.filename}")
            self.stdout.write(f"chunks: {doc.chunk_count}")
            self.stdout.write(f"Ingested: {doc.ingested_at.strftime('%Y-%m-%d %H:%M')}")
            self.stdout.write(f"Path: {doc.filename}")
            self.stdout.write("")

        # show total chunks in vector store
        stats = get_stats()
        self.stdout.write("-" * 50)
        self.stdout.write(f"Total chunks in vector store: {stats['total_chunks']}")

        
    def clear_documents(self):
        '''Wipe all vectors and DB records'''
        count = IngestedDocument.objects.count()

        if count == 0:
            self.stdout.write(self.style.WARNING("Nothing to clear."))
            return
        

        # Delete from ChromaDB
        delete_collection()

        # Delete from Django DB
        IngestedDocument.objects.all().delete()

        self.stdout.write(
            self.style.SUCCESS(
                f"Cleared {count} documensts from vector store  "
            )
        )

