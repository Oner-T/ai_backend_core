"""
Promote 👍 UserFeedback entries to GroundTruthEntry.

Usage:
    python manage.py promote_feedback          # interactive, confirms before each
    python manage.py promote_feedback --all    # promote all without prompting
    python manage.py promote_feedback --list   # just list candidates, don't promote
"""

from django.core.management.base import BaseCommand
from intelligence.models import UserFeedback, GroundTruthEntry


class Command(BaseCommand):
    help = "Promote 👍 user feedback entries to ground truth"

    def add_arguments(self, parser):
        parser.add_argument('--all',  action='store_true', help='Promote all without confirmation')
        parser.add_argument('--list', action='store_true', help='List candidates only, do not promote')

    def handle(self, *args, **options):
        already_promoted_ids = set(
            GroundTruthEntry.objects
            .filter(promoted_from__isnull=False)
            .values_list('promoted_from_id', flat=True)
        )
        candidates = (
            UserFeedback.objects
            .filter(rating='good')
            .exclude(id__in=already_promoted_ids)
            .order_by('created_at')
        )

        if not candidates.exists():
            self.stdout.write("No new 👍 feedback to promote.")
            return

        self.stdout.write(f"\nFound {candidates.count()} promotable entry/entries:\n")
        self.stdout.write(f"{'─' * 70}")

        for fb in candidates:
            source_labels = [
                f"{s.get('document_name', '?')} Madde {s.get('madde', '?')}"
                for s in fb.sources if s.get('madde')
            ]
            self.stdout.write(f"  ID {fb.pk}  |  [{', '.join(source_labels) or 'no sources'}]")
            self.stdout.write(f"  Q: {fb.question}")
            self.stdout.write(f"  A: {fb.answer[:120]}{'…' if len(fb.answer) > 120 else ''}")
            self.stdout.write(f"{'─' * 70}")

        if options['list']:
            return

        if not options['all']:
            confirm = input("\nPromote all listed entries? [y/N] ").strip().lower()
            if confirm != 'y':
                self.stdout.write("Aborted.")
                return

        promoted = 0
        skipped  = 0

        for fb in candidates:
            if GroundTruthEntry.objects.filter(question=fb.question).exists():
                self.stdout.write(f"  SKIP (duplicate question): {fb.question[:60]}")
                skipped += 1
                continue

            GroundTruthEntry.objects.create(
                question         = fb.question,
                expected_answer  = fb.answer,
                expected_sources = [
                    {"document_name": s["document_name"], "madde": s["madde"]}
                    for s in fb.sources
                    if s.get("madde") is not None and s.get("document_name")
                ],
                scenario         = 'user_validated',
                promoted_from    = fb,
            )
            promoted += 1

        self.stdout.write(f"\n✅ Promoted {promoted} entries.")
        if skipped:
            self.stdout.write(f"   Skipped {skipped} duplicate(s).")
