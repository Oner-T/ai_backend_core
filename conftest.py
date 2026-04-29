import django
from django.test.utils import setup_test_environment


def pytest_configure(config):
    pass


def pytest_sessionstart(session):
    pass


# Enable pgvector extension before Django creates the test database tables
from django.db.backends.signals import connection_created


def enable_pgvector(sender, connection, **kwargs):
    if connection.vendor == "postgresql":
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


connection_created.connect(enable_pgvector)
