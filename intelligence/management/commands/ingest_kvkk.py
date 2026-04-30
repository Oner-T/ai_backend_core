from django.core.management.base import BaseCommand
from intelligence.services import ingest_all_documents


class Command(BaseCommand):
    help = "Ingest legal PDFs: parse structure, generate embeddings, save chunks to database"

    def add_arguments(self, parser):
        parser.add_argument(
            '--regime',
            choices=['tr', 'eu', 'all'],
            default='all',
            help="Which regime to ingest: 'tr' (KVKK), 'eu' (GDPR/ePrivacy/AI Act), or 'all' (default)",
        )

    def handle(self, *args, **options):
        ingest_all_documents(regime=options['regime'])
