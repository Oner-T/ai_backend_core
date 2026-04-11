from django.core.management.base import BaseCommand
from intelligence.services import ingest_kvkk_document


class Command(BaseCommand):
    help = "Ingest KVKK.pdf: parse structure, generate embeddings, save chunks to database"

    def handle(self, *args, **options):
        ingest_kvkk_document()
