from django.contrib import admin
from intelligence.models import UserFeedback, GroundTruthEntry, EvaluationRun, EvaluationResult


# ── UserFeedback ─────────────────────────────────────────────────────────────

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display   = ('rating', 'short_question', 'created_at', 'has_comment', 'is_promoted')
    list_filter    = ('rating', 'created_at')
    search_fields  = ('question', 'answer', 'comment')
    readonly_fields = ('question', 'answer', 'sources', 'rating', 'created_at')
    ordering       = ('-created_at',)

    def short_question(self, obj):
        return obj.question[:80]
    short_question.short_description = 'Question'

    def has_comment(self, obj):
        return bool(obj.comment)
    has_comment.boolean = True
    has_comment.short_description = 'Comment?'

    def is_promoted(self, obj):
        return obj.ground_truth.exists()
    is_promoted.boolean = True
    is_promoted.short_description = 'Promoted?'


# ── GroundTruthEntry ─────────────────────────────────────────────────────────

@admin.register(GroundTruthEntry)
class GroundTruthEntryAdmin(admin.ModelAdmin):
    list_display   = ('short_question', 'scenario', 'source_count', 'created_at')
    list_filter    = ('scenario', 'created_at')
    search_fields  = ('question', 'expected_answer')
    readonly_fields = ('promoted_from', 'created_at')
    ordering       = ('-created_at',)

    def short_question(self, obj):
        return obj.question[:80]
    short_question.short_description = 'Question'

    def source_count(self, obj):
        return len(obj.expected_sources)
    source_count.short_description = 'Expected sources'


# ── EvaluationRun ────────────────────────────────────────────────────────────

class EvaluationResultInline(admin.TabularInline):
    model        = EvaluationResult
    extra        = 0
    can_delete   = False
    readonly_fields = ('entry', 'retrieval_recall', 'retrieval_hit', 'citation_hit', 'passed')
    fields       = ('entry', 'passed', 'retrieval_recall', 'retrieval_hit', 'citation_hit')
    ordering     = ('passed', 'entry__question')

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display   = ('run_at', 'llm_model', 'fmt_pass_rate', 'fmt_recall', 'fmt_citation', 'total_questions', 'git_revision')
    readonly_fields = ('run_at', 'git_revision', 'llm_model', 'total_questions',
                       'pass_rate', 'retrieval_recall', 'citation_accuracy')
    ordering       = ('-run_at',)
    inlines        = [EvaluationResultInline]

    def fmt_pass_rate(self, obj):
        return f"{obj.pass_rate:.0%}"
    fmt_pass_rate.short_description = 'Pass rate'

    def fmt_recall(self, obj):
        return f"{obj.retrieval_recall:.0%}"
    fmt_recall.short_description = 'Retrieval recall'

    def fmt_citation(self, obj):
        return f"{obj.citation_accuracy:.0%}"
    fmt_citation.short_description = 'Citation acc.'
