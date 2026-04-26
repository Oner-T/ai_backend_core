"""
Run evaluation against all GroundTruthEntry rows and save results.

Usage:
    python manage.py run_evaluation
    python manage.py run_evaluation --scenario user_validated
    python manage.py run_evaluation --dry-run   # compute metrics, don't save to DB

Metrics computed per question:
    retrieval_hit    — at least one expected (document, madde) pair was retrieved
    retrieval_recall — fraction of expected sources that were retrieved
    citation_hit     — answer text cites at least one expected madde number
    passed           — both retrieval_hit AND citation_hit

Aggregate (saved to EvaluationRun):
    pass_rate         — fraction of questions that passed
    retrieval_recall  — average retrieval recall across all questions
    citation_accuracy — fraction of questions with citation_hit
"""

import re
import subprocess

from django.core.management.base import BaseCommand

from intelligence.models import GroundTruthEntry, EvaluationRun, EvaluationResult
from intelligence.services import query_kvkk, ANSWER_MODEL

DIVIDER  = "─" * 70
BOLD_DIV = "═" * 70


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


class Command(BaseCommand):
    help = "Evaluate the RAG pipeline against ground truth entries"

    def add_arguments(self, parser):
        parser.add_argument(
            '--scenario', type=str, default=None,
            help='Filter by scenario (e.g. user_validated, cross_doc, out_of_scope)'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Compute and print metrics without saving to DB'
        )

    def handle(self, *args, **options):
        qs = GroundTruthEntry.objects.all()
        if options['scenario']:
            qs = qs.filter(scenario=options['scenario'])

        entries = list(qs.order_by('created_at'))
        if not entries:
            self.stdout.write("No ground truth entries found. Run promote_feedback first.")
            return

        git_rev  = _git_revision()
        dry_run  = options['dry_run']

        self.stdout.write(f"\n{BOLD_DIV}")
        self.stdout.write(f"  Evaluation Run  —  {len(entries)} questions  |  model: {ANSWER_MODEL}")
        self.stdout.write(f"  git: {git_rev}{'  [DRY RUN]' if dry_run else ''}")
        self.stdout.write(f"{BOLD_DIV}\n")

        pending_results: list[EvaluationResult] = []

        for idx, entry in enumerate(entries, 1):
            self.stdout.write(f"[{idx:02d}/{len(entries)}] {entry.question[:65]}")

            result = query_kvkk(entry.question, history=[])

            # ── Retrieval metrics ────────────────────────────────────────────
            expected_pairs = {
                (s["document_name"], s["madde"])
                for s in entry.expected_sources
                if s.get("document_name") and s.get("madde") is not None
            }
            returned_pairs = {
                (s["document_name"], s["madde"])
                for s in result["sources"]
                if s.get("document_name") and s.get("madde") is not None
            }

            if expected_pairs:
                overlap        = expected_pairs & returned_pairs
                ret_recall     = len(overlap) / len(expected_pairs)
                ret_hit        = bool(overlap)
            else:
                # out_of_scope: pass only if pipeline returned no sources
                ret_hit    = len(returned_pairs) == 0
                ret_recall = 1.0 if ret_hit else 0.0

            # ── Citation metrics ─────────────────────────────────────────────
            cited_maddes    = {
                int(m) for m in re.findall(r'Madde\s+(\d+)', result["answer"], re.IGNORECASE)
            }
            expected_maddes = {
                s["madde"] for s in entry.expected_sources if s.get("madde") is not None
            }
            cit_hit = bool(expected_maddes & cited_maddes) if expected_maddes else True

            passed = ret_hit and cit_hit
            icon   = "✅" if passed else "❌"

            self.stdout.write(
                f"  {icon}  retrieval={ret_recall:.0%}"
                f"  citation={'✓' if cit_hit else '✗'}"
                f"  sources_returned={len(result['sources'])}"
            )
            if not ret_hit and expected_pairs:
                missing = expected_pairs - returned_pairs
                self.stdout.write(f"     missing: {[f'{d} Madde {m}' for d, m in missing]}")

            pending_results.append(EvaluationResult(
                entry             = entry,
                generated_answer  = result["answer"],
                retrieved_sources = result["sources"],
                retrieval_recall  = ret_recall,
                retrieval_hit     = ret_hit,
                citation_hit      = cit_hit,
                passed            = passed,
            ))

        # ── Aggregate metrics ────────────────────────────────────────────────
        n             = len(pending_results)
        pass_rate     = sum(1 for r in pending_results if r.passed) / n
        avg_recall    = sum(r.retrieval_recall for r in pending_results) / n
        citation_acc  = sum(1 for r in pending_results if r.citation_hit) / n

        self.stdout.write(f"\n{BOLD_DIV}")
        self.stdout.write("  RESULTS")
        self.stdout.write(BOLD_DIV)
        self.stdout.write(f"  Pass rate:         {pass_rate:.0%}  ({sum(1 for r in pending_results if r.passed)}/{n})")
        self.stdout.write(f"  Retrieval recall:  {avg_recall:.0%}")
        self.stdout.write(f"  Citation accuracy: {citation_acc:.0%}")

        # ── Trend comparison ─────────────────────────────────────────────────
        prev = EvaluationRun.objects.first()
        if prev:
            def delta(new, old):
                d = new - old
                return f"({'+'if d >= 0 else ''}{d:+.0%})"
            self.stdout.write(f"\n  vs previous run ({prev.run_at:%Y-%m-%d %H:%M}):")
            self.stdout.write(f"    Pass rate:        {prev.pass_rate:.0%} → {pass_rate:.0%}  {delta(pass_rate, prev.pass_rate)}")
            self.stdout.write(f"    Retrieval recall: {prev.retrieval_recall:.0%} → {avg_recall:.0%}  {delta(avg_recall, prev.retrieval_recall)}")
            self.stdout.write(f"    Citation acc:     {prev.citation_accuracy:.0%} → {citation_acc:.0%}  {delta(citation_acc, prev.citation_accuracy)}")

        if dry_run:
            self.stdout.write(f"\n  [DRY RUN] — results not saved.")
        else:
            run = EvaluationRun.objects.create(
                git_revision      = git_rev,
                llm_model         = ANSWER_MODEL,
                total_questions   = n,
                pass_rate         = pass_rate,
                retrieval_recall  = avg_recall,
                citation_accuracy = citation_acc,
            )
            for r in pending_results:
                r.run = run
            EvaluationResult.objects.bulk_create(pending_results)
            self.stdout.write(f"\n  Saved as EvaluationRun #{run.pk}.")

        self.stdout.write(f"{BOLD_DIV}\n")
