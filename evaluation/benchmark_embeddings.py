"""
Embedding Model Benchmark for KVKK RAG
=======================================
Tests multiple free multilingual embedding models against our actual KVKK chunks
using golden dataset queries. Reports MRR and Recall@k per model.

Usage:
    source venv/bin/activate
    python evaluation/benchmark_embeddings.py

Models tested (all free, no API key needed):
    1. paraphrase-multilingual-MiniLM-L12-v2  — current model (384d, fast)
    2. paraphrase-multilingual-mpnet-base-v2  — stronger sibling (768d)
    3. intfloat/multilingual-e5-base          — instruction-tuned (768d)
    4. intfloat/multilingual-e5-large         — best E5 (1024d, slower)
    5. BAAI/bge-m3                            — state-of-art multilingual (1024d)
"""

import json
import os
import sys
import time
import numpy as np
import django

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_api.settings")
django.setup()

from intelligence.models import DocumentChunk
from sentence_transformers import SentenceTransformer

# ── Test cases ──────────────────────────────────────────────────────────────
# Mix of easy (term matches) and hard (semantic mismatch) cases
TEST_CASES = [
    # Hard — vernacular vs legal phrasing
    {"query": "Veri ihlali durumunda ne yapılmalıdır?",        "expected_maddes": [12]},
    {"query": "Özel nitelikli kişisel veriler nelerdir?",      "expected_maddes": [6]},
    {"query": "Açık rıza olmadan veri işlenebilir mi?",        "expected_maddes": [5]},
    {"query": "Kişisel verileri yurt dışına çıkarabilir miyim?", "expected_maddes": [9]},
    # Medium
    {"query": "Veri saklama süresi ne kadar olmalı?",          "expected_maddes": [7]},
    {"query": "Kurul kimlerden oluşur?",                       "expected_maddes": [21]},
    {"query": "İlgili kişi haklarını nasıl kullanır?",         "expected_maddes": [11, 13]},
    {"query": "Veri işlemenin hukuki dayanakları nelerdir?",   "expected_maddes": [5]},
    # Easy — explicit term overlap
    {"query": "Kişisel veri nedir?",                           "expected_maddes": [3]},
    {"query": "Veri sorumlusu kimdir?",                        "expected_maddes": [3]},
    {"query": "Kişisel verilerin işlenme şartları nelerdir?",  "expected_maddes": [5]},
    {"query": "Kurumun görev ve yetkileri nelerdir?",          "expected_maddes": [22]},
]

MODELS = [
    {
        "name": "MiniLM-L12 (current)",
        "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": None,       # no instruction prefix needed
        "dims": 384,
    },
    {
        "name": "mpnet-base",
        "model_id": "paraphrase-multilingual-mpnet-base-v2",
        "prefix": None,
        "dims": 768,
    },
    {
        "name": "e5-base",
        "model_id": "intfloat/multilingual-e5-base",
        "prefix": "query: ",  # E5 requires "query: " / "passage: " prefixes
        "dims": 768,
    },
    {
        "name": "e5-large",
        "model_id": "intfloat/multilingual-e5-large",
        "prefix": "query: ",
        "dims": 1024,
    },
    {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "prefix": None,
        "dims": 1024,
    },
]

TOP_K_VALUES = [1, 3, 5, 8]
DIVIDER = "─" * 72
BOLD_DIV = "═" * 72


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def reciprocal_rank(ranked_maddes: list[int | None], expected: list[int]) -> float:
    for rank, madde in enumerate(ranked_maddes, start=1):
        if madde in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_maddes: list[int | None], expected: list[int], k: int) -> float:
    top_k_maddes = set(ranked_maddes[:k])
    hit = any(m in top_k_maddes for m in expected)
    return 1.0 if hit else 0.0


def run_benchmark():
    print(f"\n{BOLD_DIV}")
    print("  KVKK Embedding Model Benchmark")
    print(f"{BOLD_DIV}\n")

    # Load all chunks from DB once
    print("📦 Loading chunks from database...")
    chunks = list(DocumentChunk.objects.select_related('section').all())
    print(f"   {len(chunks)} chunks loaded\n")

    if not chunks:
        print("❌ No chunks found — run `python manage.py ingest_kvkk` first.")
        sys.exit(1)

    # Strip headers from chunk content for content-only embedding (mirrors ingestion fix)
    chunk_texts_raw = [
        c.content.split('\n', 1)[1].strip() if '\n' in c.content else c.content
        for c in chunks
    ]
    chunk_maddes = [c.madde for c in chunks]

    summary_rows = []

    for model_cfg in MODELS:
        print(f"{DIVIDER}")
        print(f"🔬 Model: {model_cfg['name']}  ({model_cfg['model_id']})")
        print(f"   dims={model_cfg['dims']}")

        t0 = time.time()
        try:
            model = SentenceTransformer(model_cfg["model_id"])
        except Exception as e:
            print(f"   ❌ Failed to load: {e}\n")
            continue

        # Embed all chunks (passage prefix for E5 models)
        passage_prefix = model_cfg["prefix"].replace("query: ", "passage: ") if model_cfg["prefix"] else ""
        chunk_inputs = [f"{passage_prefix}{t}" for t in chunk_texts_raw]

        print(f"   Embedding {len(chunks)} chunks...", end=" ", flush=True)
        chunk_embeddings = model.encode(chunk_inputs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        print(f"done ({time.time()-t0:.1f}s)")

        # Per-query metrics
        rr_scores = []
        recall_scores = {k: [] for k in TOP_K_VALUES}
        hard_misses = []

        for tc in TEST_CASES:
            query = tc["query"]
            expected = tc["expected_maddes"]

            q_input = f"{model_cfg['prefix']}{query}" if model_cfg["prefix"] else query
            q_emb = model.encode([q_input], normalize_embeddings=True)[0]

            sims = [cosine_similarity(q_emb, c_emb) for c_emb in chunk_embeddings]
            ranked_indices = np.argsort(sims)[::-1]
            ranked_maddes = [chunk_maddes[i] for i in ranked_indices]

            rr = reciprocal_rank(ranked_maddes, expected)
            rr_scores.append(rr)

            for k in TOP_K_VALUES:
                recall_scores[k].append(recall_at_k(ranked_maddes, expected, k))

            # Find rank of first correct hit for display
            first_hit_rank = next(
                (r + 1 for r, m in enumerate(ranked_maddes) if m in expected), None
            )
            hit_marker = "✅" if (first_hit_rank or 99) <= 5 else "❌"
            if (first_hit_rank or 99) > 5:
                hard_misses.append(f"    ❌ '{query[:55]}' → first hit at rank {first_hit_rank} (expected Madde {expected})")

        mrr = float(np.mean(rr_scores))
        elapsed = time.time() - t0

        print(f"\n   MRR:        {mrr:.3f}")
        for k in TOP_K_VALUES:
            r = float(np.mean(recall_scores[k]))
            print(f"   Recall@{k}:  {r:.3f}  ({sum(1 for x in recall_scores[k] if x)}/{len(TEST_CASES)} hits)")

        if hard_misses:
            print(f"\n   Hard misses:")
            for m in hard_misses:
                print(m)

        print(f"\n   Total time: {elapsed:.1f}s\n")

        summary_rows.append({
            "name": model_cfg["name"],
            "dims": model_cfg["dims"],
            "mrr": mrr,
            "recall@5": float(np.mean(recall_scores[5])),
            "recall@3": float(np.mean(recall_scores[3])),
            "time_s": round(elapsed, 1),
        })

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{BOLD_DIV}")
    print("  SUMMARY")
    print(f"{BOLD_DIV}")
    print(f"  {'Model':<22} {'Dims':>5}  {'MRR':>6}  {'R@3':>6}  {'R@5':>6}  {'Time':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
    for r in sorted(summary_rows, key=lambda x: x["mrr"], reverse=True):
        marker = " ← best" if r == max(summary_rows, key=lambda x: x["mrr"]) else ""
        print(
            f"  {r['name']:<22} {r['dims']:>5}  {r['mrr']:>6.3f}  "
            f"{r['recall@3']:>6.3f}  {r['recall@5']:>6.3f}  {r['time_s']:>6.1f}s{marker}"
        )
    print(f"{BOLD_DIV}\n")


if __name__ == "__main__":
    run_benchmark()
