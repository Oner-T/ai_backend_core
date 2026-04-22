from django.core.management.base import BaseCommand
from intelligence.services import ingest_all_documents


class Command(BaseCommand):
    help = "Ingest all KVKK-related PDFs: parse structure, generate embeddings, save chunks to database"

    def handle(self, *args, **options):
        ingest_all_documents()
