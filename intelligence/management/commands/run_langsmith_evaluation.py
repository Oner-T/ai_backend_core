"""
Push GroundTruthEntry rows to LangSmith as a Dataset and run a scored evaluation.

Metrics per question:
    retrieval_recall  — fraction of expected (document_name, madde) pairs retrieved
    retrieval_hit     — 1.0 if at least one expected pair retrieved, else 0.0
    citation_hit      — 1.0 if answer cites at least one expected madde, else 0.0
    ndcg              — nDCG@k for the ranked retrieval list
    mrr               — Mean Reciprocal Rank

Usage:
    python manage.py run_langsmith_evaluation
    python manage.py run_langsmith_evaluation --scenario user_validated
    python manage.py run_langsmith_evaluation --dataset-name "kvkk-eval-v2"
"""

import math
import re

from django.core.management.base import BaseCommand

from intelligence.models import GroundTruthEntry
from intelligence.services import query_kvkk

try:
    from langsmith import Client
    from langsmith.evaluation import evaluate
except ImportError:
    Client = None


def _ndcg(expected_pairs: set, returned_pairs: list, k: int = 8) -> float:
    """nDCG@k: returned_pairs is an ordered list from rank-1 to rank-k."""
    if not expected_pairs:
        return 1.0
    gains = [1.0 if p in expected_pairs else 0.0 for p in returned_pairs[:k]]
    dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_pairs), k)))
    return dcg / ideal if ideal else 0.0


def _mrr(expected_pairs: set, returned_pairs: list) -> float:
    """Mean Reciprocal Rank — rank of first relevant result."""
    for i, p in enumerate(returned_pairs):
        if p in expected_pairs:
            return 1.0 / (i + 1)
    return 0.0


def _run_pipeline(inputs: dict) -> dict:
    """Target function called by LangSmith evaluate()."""
    result = query_kvkk(inputs["question"], history=[])
    return {
        "answer":  result["answer"],
        "sources": result["sources"],
    }


def _make_evaluator(entry: GroundTruthEntry):
    """Returns a callable evaluator for a single ground truth entry."""
    expected_pairs  = [(s["document_name"], s["madde"]) for s in entry.expected_sources]
    expected_set    = set(map(tuple, expected_pairs))
    expected_maddes = {s["madde"] for s in entry.expected_sources if s.get("madde") is not None}

    def evaluator(run, example):
        output = run.outputs or {}
        sources = output.get("sources", [])

        returned_ordered = [
            (s["document_name"], s["madde"])
            for s in sources
            if s.get("document_name") and s.get("madde") is not None
        ]
        returned_set = set(returned_ordered)

        if expected_set:
            overlap  = expected_set & returned_set
            recall   = len(overlap) / len(expected_set)
            ret_hit  = float(bool(overlap))
            ndcg_val = _ndcg(expected_set, returned_ordered)
            mrr_val  = _mrr(expected_set, returned_ordered)
        else:
            # out_of_scope: pass only if pipeline returned no sources
            ret_hit  = float(len(returned_set) == 0)
            recall   = ret_hit
            ndcg_val = ret_hit
            mrr_val  = ret_hit

        answer = output.get("answer", "")
        cited  = {int(m) for m in re.findall(r'Madde\s+(\d+)', answer, re.IGNORECASE)}
        cit_hit = float(bool(expected_maddes & cited)) if expected_maddes else 1.0

        passed = float(ret_hit == 1.0 and cit_hit == 1.0)

        return {
            "key":   "passed",
            "score": passed,
            "metadata": {
                "retrieval_recall": recall,
                "retrieval_hit":    ret_hit,
                "citation_hit":     cit_hit,
                "ndcg":             ndcg_val,
                "mrr":              mrr_val,
            },
        }

    return evaluator


DIVIDER  = "─" * 70
BOLD_DIV = "═" * 70


class Command(BaseCommand):
    help = "Run LangSmith evaluation with nDCG + MRR against GroundTruthEntry rows"

    def add_arguments(self, parser):
        parser.add_argument(
            '--scenario', type=str, default=None,
            help='Filter by scenario'
        )
        parser.add_argument(
            '--dataset-name', type=str, default='kvkk-ground-truth',
            help='LangSmith dataset name (created or updated automatically)'
        )

    def handle(self, *args, **options):
        if Client is None:
            self.stderr.write("langsmith package not installed. Run: pip install langsmith")
            return

        scenario = options['scenario']
        dataset_name = options['dataset_name']

        qs = GroundTruthEntry.objects.all()
        if scenario:
            qs = qs.filter(scenario=scenario)
        entries = list(qs.order_by('created_at'))

        if not entries:
            self.stdout.write("No ground truth entries found.")
            return

        self.stdout.write(f"\n{BOLD_DIV}")
        self.stdout.write(f"  LangSmith Evaluation  —  {len(entries)} questions")
        self.stdout.write(f"  Dataset: {dataset_name}")
        self.stdout.write(f"{BOLD_DIV}\n")

        client = Client()

        # ── Upsert LangSmith dataset ─────────────────────────────────────────
        existing_datasets = {ds.name: ds for ds in client.list_datasets()}

        if dataset_name in existing_datasets:
            dataset = existing_datasets[dataset_name]
            self.stdout.write(f"  Using existing dataset: {dataset.name} (id={dataset.id})")
            # Delete old examples so we push fresh ones
            for ex in client.list_examples(dataset_id=dataset.id):
                client.delete_example(ex.id)
            self.stdout.write(f"  Cleared old examples.")
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)
            self.stdout.write(f"  Created new dataset: {dataset.name}")

        # ── Push examples ────────────────────────────────────────────────────
        client.create_examples(
            inputs   = [{"question": e.question} for e in entries],
            outputs  = [{"expected_sources": e.expected_sources} for e in entries],
            dataset_id = dataset.id,
        )
        self.stdout.write(f"  Pushed {len(entries)} examples.\n")

        # ── Build per-entry evaluators ───────────────────────────────────────
        evaluators = [_make_evaluator(e) for e in entries]

        # LangSmith evaluate() calls each evaluator for each run.
        # We pass a list of evaluators; LangSmith calls evaluators[i] for example i.
        # Since evaluate() doesn't natively support per-example evaluators,
        # we build a single dispatcher evaluator that looks up by question.
        entry_map = {e.question: e for e in entries}

        def dispatcher_evaluator(run, example):
            question = (example.inputs or {}).get("question", "")
            entry    = entry_map.get(question)
            if entry is None:
                return {"key": "passed", "score": 0.0}
            return _make_evaluator(entry)(run, example)

        # ── Run evaluation ───────────────────────────────────────────────────
        self.stdout.write("  Running pipeline against dataset (this may take a few minutes)...\n")

        results = evaluate(
            _run_pipeline,
            data        = dataset_name,
            evaluators  = [dispatcher_evaluator],
            experiment_prefix = "kvkk-rag",
            metadata    = {"dataset": dataset_name, "scenario": scenario or "all"},
        )

        # ── Aggregate and print ──────────────────────────────────────────────
        scores      = [r["evaluation_results"]["results"][0].score for r in results._results]
        metadatas   = [r["evaluation_results"]["results"][0].metadata or {} for r in results._results]

        n           = len(scores)
        pass_rate   = sum(scores) / n
        avg_recall  = sum(m.get("retrieval_recall", 0) for m in metadatas) / n
        avg_ndcg    = sum(m.get("ndcg", 0) for m in metadatas) / n
        avg_mrr     = sum(m.get("mrr", 0) for m in metadatas) / n
        cit_acc     = sum(m.get("citation_hit", 0) for m in metadatas) / n

        self.stdout.write(f"\n{BOLD_DIV}")
        self.stdout.write("  RESULTS")
        self.stdout.write(BOLD_DIV)
        self.stdout.write(f"  Pass rate:         {pass_rate:.0%}  ({sum(1 for s in scores if s == 1.0)}/{n})")
        self.stdout.write(f"  Retrieval recall:  {avg_recall:.0%}")
        self.stdout.write(f"  Citation accuracy: {cit_acc:.0%}")
        self.stdout.write(f"  nDCG@8:            {avg_ndcg:.3f}")
        self.stdout.write(f"  MRR:               {avg_mrr:.3f}")
        self.stdout.write(f"\n  View in LangSmith: https://smith.langchain.com")
        self.stdout.write(f"  Project: {options.get('dataset_name', dataset_name)}")
        self.stdout.write(f"{BOLD_DIV}\n")
