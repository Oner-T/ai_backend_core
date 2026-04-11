from django.db import models
from pgvector.django import VectorField # <-- 1. Add this import

class DocumentSection(models.Model):
    number = models.IntegerField(unique=True, help_text="e.g., 1 for BİRİNCİ BÖLÜM")

    def __str__(self):
        return str(self.number)

class DocumentChunk(models.Model):
    section = models.ForeignKey(DocumentSection, on_delete=models.CASCADE, related_name="chunks", null=True)
    madde = models.IntegerField(null=True, blank=True)
    document_name = models.CharField(max_length=255, default="KVKK.pdf")
    chunk_index = models.IntegerField()
    content = models.TextField()
    
    # UPDATE THIS LINE: Change dimensions from 1536 to 384
    embedding = VectorField(dimensions=384, null=True, blank=True, help_text="The AI mathematical representation of the text")

    def __str__(self):
        return f"{self.section.number if self.section else 'Genel'} - {self.madde} - Row {self.chunk_index}"