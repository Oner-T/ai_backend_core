from django.db import models
from pgvector.django import VectorField

class DocumentSection(models.Model):
    number = models.IntegerField(unique=True, help_text="e.g., 1 for BİRİNCİ BÖLÜM")

    def __str__(self):
        return str(self.number)

class DocumentChunk(models.Model):
    section = models.ForeignKey(DocumentSection, on_delete=models.CASCADE, related_name="chunks", null=True)
    madde = models.IntegerField(null=True, blank=True)
    madde_title = models.CharField(max_length=255, null=True, blank=True)
    document_name = models.CharField(max_length=255, default="KVKK.pdf")
    chunk_index = models.IntegerField()
    content = models.TextField()
    embedding = VectorField(dimensions=1024, null=True, blank=True, help_text="The AI mathematical representation of the text")

    def __str__(self):
        return f"{self.section.number if self.section else 'Genel'} - {self.madde} - Row {self.chunk_index}"


class UserFeedback(models.Model):
    RATING_CHOICES = [('good', '👍 Good'), ('bad', '👎 Bad')]

    question   = models.TextField()
    answer     = models.TextField()
    sources    = models.JSONField(default=list)
    rating     = models.CharField(max_length=10, choices=RATING_CHOICES)
    comment    = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.rating}] {self.question[:60]}"


class GroundTruthEntry(models.Model):
    SCENARIO_CHOICES = [
        ('definition',    'Definition'),
        ('procedural',    'Procedural'),
        ('cross_doc',     'Cross-document'),
        ('specific_madde','Specific Madde'),
        ('secondary_law', 'Secondary Law'),
        ('out_of_scope',  'Out of Scope'),
        ('user_validated','User Validated'),
    ]

    question         = models.TextField(unique=True)
    expected_answer  = models.TextField(blank=True)
    # [{document_name: str, madde: int}, ...]
    expected_sources = models.JSONField(default=list)
    scenario         = models.CharField(max_length=50, choices=SCENARIO_CHOICES, default='user_validated')
    promoted_from    = models.ForeignKey(
        UserFeedback, null=True, blank=True, on_delete=models.SET_NULL, related_name='ground_truth'
    )
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.question[:80]


class EvaluationRun(models.Model):
    run_at            = models.DateTimeField(auto_now_add=True)
    git_revision      = models.CharField(max_length=50, blank=True)
    llm_model         = models.CharField(max_length=100)
    total_questions   = models.IntegerField()
    pass_rate         = models.FloatField()
    retrieval_recall  = models.FloatField()
    citation_accuracy = models.FloatField()

    class Meta:
        ordering = ['-run_at']

    def __str__(self):
        return f"Run {self.run_at:%Y-%m-%d %H:%M} — {self.pass_rate:.0%} pass ({self.total_questions}q)"


class EvaluationResult(models.Model):
    run               = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name='results')
    entry             = models.ForeignKey(GroundTruthEntry, on_delete=models.CASCADE, related_name='results')
    generated_answer  = models.TextField()
    retrieved_sources = models.JSONField(default=list)
    retrieval_recall  = models.FloatField()
    retrieval_hit     = models.BooleanField()
    citation_hit      = models.BooleanField()
    passed            = models.BooleanField()

    def __str__(self):
        return f"{'✅' if self.passed else '❌'} {self.entry.question[:60]}"