"""
Ablation Test — KVKK RAG Pipeline
===================================
Isolates the contribution of each change by running controlled configs.

Retrieval ablation (free, automatic):
    Tests embedding model, header vs content-only, and Mülga filter independently.

LLM ablation (optional, costs tokens):
    Compares llama3.2 vs Gemini on the same retrieved chunks.
    Run with: python evaluation/ablation_test.py --llm

Usage:
    source venv/bin/activate
    python evaluation/ablation_test.py           # retrieval only
    python evaluation/ablation_test.py --llm     # retrieval + LLM comparison
"""

import os, re, sys, time, json
import numpy as np
import django

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_api.settings")
django.setup()

from intelligence.models import DocumentChunk
from sentence_transformers import SentenceTransformer

RUN_LLM = "--llm" in sys.argv

# ── Retrieval configs ────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name":         "A  MiniLM + with headers",
        "model_id":     "paraphrase-multilingual-MiniLM-L12-v2",
        "strip_header": False,
        "filter_mulga": False,
        "q_prefix":     "",
        "p_prefix":     "",
    },
    {
        "name":         "B  MiniLM + content-only",
        "model_id":     "paraphrase-multilingual-MiniLM-L12-v2",
        "strip_header": True,
        "filter_mulga": False,
        "q_prefix":     "",
        "p_prefix":     "",
    },
    {
        "name":         "C  e5-large + with headers",
        "model_id":     "intfloat/multilingual-e5-large",
        "strip_header": False,
        "filter_mulga": False,
        "q_prefix":     "query: ",
        "p_prefix":     "passage: ",
    },
    {
        "name":         "D  e5-large + content-only  (no Mülga filter)",
        "model_id":     "intfloat/multilingual-e5-large",
        "strip_header": True,
        "filter_mulga": False,
        "q_prefix":     "query: ",
        "p_prefix":     "passage: ",
    },
    {
        "name":         "E  e5-large + content-only + Mülga filter  ← current",
        "model_id":     "intfloat/multilingual-e5-large",
        "strip_header": True,
        "filter_mulga": True,
        "q_prefix":     "query: ",
        "p_prefix":     "passage: ",
    },
]

# ── Test cases (from golden dataset — retrieval focused) ─────────────────────
TEST_CASES = [
    # Hard — semantic gap between query and chunk
    {"query": "Veri ihlali durumunda ne yapılmalıdır?",          "expected_maddes": [12]},
    {"query": "Açık rıza şartları nelerdir?",                    "expected_maddes": [3]},
    {"query": "Özel nitelikli kişisel veriler nelerdir?",        "expected_maddes": [6]},
    {"query": "Açık rıza olmadan veri işlenebilir mi?",          "expected_maddes": [5]},
    {"query": "Kişisel verileri yurt dışına çıkarabilir miyim?", "expected_maddes": [9]},
    # Medium
    {"query": "Veri saklama süresi ne kadar olmalı?",            "expected_maddes": [7]},
    {"query": "İlgili kişi haklarını nasıl kullanır?",           "expected_maddes": [11, 13]},
    {"query": "Veri işlemenin hukuki dayanakları nelerdir?",     "expected_maddes": [5]},
    {"query": "Kurul kimlerden oluşur?",                         "expected_maddes": [21]},
    # Easy — direct term overlap
    {"query": "Kişisel veri nedir?",                             "expected_maddes": [3]},
    {"query": "Veri sorumlusu kimdir?",                          "expected_maddes": [3]},
    {"query": "Kişisel verilerin işlenme şartları nelerdir?",    "expected_maddes": [5]},
    {"query": "Kurumun görev ve yetkileri nelerdir?",            "expected_maddes": [22]},
    # Mülga filter relevant — these had deleted chunks polluting results
    {"query": "Özel nitelikli kişisel verilerin işlenmesi hangi durumlarda mümkündür?", "expected_maddes": [6]},
    {"query": "Veri sorumlusunun aydınlatma yükümlülüğü nedir?", "expected_maddes": [10]},
]

TOP_K = [1, 3, 5, 8]
DIVIDER  = "─" * 72
BOLD_DIV = "═" * 72

MULGA_RE = re.compile(r'[Mm]ülga')


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def mrr(ranked_maddes, expected):
    for rank, m in enumerate(ranked_maddes, 1):
        if m in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_maddes, expected, k):
    return 1.0 if any(m in set(ranked_maddes[:k]) for m in expected) else 0.0


# ── PART 1: Retrieval ablation ───────────────────────────────────────────────
print(f"\n{BOLD_DIV}")
print("  PART 1 — Retrieval Ablation  (no API cost)")
print(f"{BOLD_DIV}\n")

all_chunks = list(DocumentChunk.objects.select_related('section').all())
print(f"Total chunks in DB: {len(all_chunks)}\n")

retrieval_summary = []
loaded_models: dict[str, SentenceTransformer] = {}

for cfg in CONFIGS:
    print(f"{DIVIDER}")
    print(f"Config: {cfg['name']}")

    # Filter Mülga if requested
    chunks = [c for c in all_chunks if not (cfg["filter_mulga"] and MULGA_RE.search(c.content))]
    print(f"  Chunks after filter: {len(chunks)}")

    # Build passage texts
    def passage_text(c):
        raw = c.content.split('\n', 1)[1].strip() if '\n' in c.content else c.content
        body = raw if cfg["strip_header"] else c.content
        return cfg["p_prefix"] + body

    passage_texts = [passage_text(c) for c in chunks]
    chunk_maddes  = [c.madde for c in chunks]

    # Load model (cache between configs)
    mid = cfg["model_id"]
    if mid not in loaded_models:
        print(f"  Loading {mid}...", end=" ", flush=True)
        t0 = time.time()
        loaded_models[mid] = SentenceTransformer(mid)
        print(f"done ({time.time()-t0:.1f}s)")
    model = loaded_models[mid]

    # Embed passages
    t0 = time.time()
    p_embs = model.encode(passage_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    embed_time = time.time() - t0
    print(f"  Embedded {len(chunks)} passages in {embed_time:.1f}s")

    # Run queries
    mrr_scores  = []
    rec_scores  = {k: [] for k in TOP_K}
    hard_misses = []

    for tc in TEST_CASES:
        q_text = cfg["q_prefix"] + tc["query"]
        q_emb  = model.encode([q_text], normalize_embeddings=True)[0]

        sims = [cosine_sim(q_emb, p) for p in p_embs]
        ranked_idx    = np.argsort(sims)[::-1]
        ranked_maddes = [chunk_maddes[i] for i in ranked_idx]

        mrr_scores.append(mrr(ranked_maddes, tc["expected_maddes"]))
        for k in TOP_K:
            rec_scores[k].append(recall_at_k(ranked_maddes, tc["expected_maddes"], k))

        first_hit = next((r+1 for r, m in enumerate(ranked_maddes) if m in tc["expected_maddes"]), None)
        if (first_hit or 99) > 5:
            hard_misses.append((tc["query"][:55], first_hit, tc["expected_maddes"]))

    mean_mrr = float(np.mean(mrr_scores))
    print(f"\n  MRR:       {mean_mrr:.3f}")
    for k in TOP_K:
        r = float(np.mean(rec_scores[k]))
        hits = sum(1 for x in rec_scores[k] if x)
        print(f"  Recall@{k}: {r:.3f}  ({hits}/{len(TEST_CASES)} hits)")

    if hard_misses:
        print(f"\n  Hard misses (rank > 5):")
        for q, rank, exp in hard_misses:
            print(f"    ❌ '{q}' → rank {rank}  (expected Madde {exp})")

    retrieval_summary.append({
        "name":    cfg["name"],
        "mrr":     mean_mrr,
        "rec@3":   float(np.mean(rec_scores[3])),
        "rec@5":   float(np.mean(rec_scores[5])),
        "misses":  len(hard_misses),
    })
    print()


# ── PART 2: LLM ablation (optional) ─────────────────────────────────────────
llm_summary = []

if RUN_LLM:
    print(f"\n{BOLD_DIV}")
    print("  PART 2 — LLM Ablation  (uses API tokens)")
    print(f"{BOLD_DIV}\n")

    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_ollama import ChatOllama
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    load_dotenv()

    SYSTEM_PROMPT = (
        "You are an expert on KVKK. Answer strictly based on the provided context. "
        "Always cite the Madde number. Reply in Turkish."
    )

    # Use current DB embeddings (e5-large + content-only) for retrieval
    from pgvector.django import CosineDistance
    from intelligence.services import get_embeddings_model

    LLM_CONFIGS = [
        {"name": "llama3.2 (local)", "llm": ChatOllama(model="llama3.2", temperature=0, timeout=60)},
        {"name": "Gemini 2.5 Flash", "llm": ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"), timeout=60
        )},
    ]

    # Use a subset of harder questions for LLM test (reduce cost)
    LLM_TEST_CASES = [
        {"query": "Özel nitelikli kişisel verilerin işlenmesi hangi durumlarda mümkündür?", "expected_maddes": [6]},
        {"query": "Veri ihlali durumunda ne yapılmalıdır?",   "expected_maddes": [12]},
        {"query": "İlgili kişinin hakları nelerdir?",         "expected_maddes": [11]},
        {"query": "Açık rıza olmadan veri işlenebilir mi?",   "expected_maddes": [5]},
        {"query": "Veri sorumlusunun yükümlülükleri nelerdir?", "expected_maddes": [10, 12]},
    ]

    emb_model = get_embeddings_model()

    for llm_cfg in LLM_CONFIGS:
        print(f"{DIVIDER}")
        print(f"LLM: {llm_cfg['name']}")
        llm = llm_cfg["llm"]

        citation_hits = 0
        total_tokens  = 0
        total_time    = 0.0

        for tc in LLM_TEST_CASES:
            q_emb = emb_model.embed_query("query: " + tc["query"])
            chunks = list(
                DocumentChunk.objects
                .exclude(content__icontains='Mülga')
                .annotate(dist=CosineDistance('embedding', q_emb))
                .order_by('dist')[:5]
            )
            context = "\n\n".join(
                f"[Madde {c.madde}]\n{c.content.split(chr(10),1)[1].strip() if chr(10) in c.content else c.content}"
                for c in chunks
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Bağlam:\n{context}\n\nSoru: {tc['query']}"),
            ]
            t0 = time.time()
            try:
                resp = llm.invoke(messages)
                elapsed = time.time() - t0
                answer  = resp.content
                usage   = resp.usage_metadata or {}
                tokens  = usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

            # Check if answer cites expected madde
            cited = any(f"Madde {m}" in answer or str(m) in answer for m in tc["expected_maddes"])
            citation_hits += int(cited)
            total_tokens  += tokens
            total_time    += elapsed

            marker = "✅" if cited else "❌"
            print(f"  {marker} '{tc['query'][:50]}'")
            print(f"     cited={cited}  tokens={tokens}  time={elapsed:.1f}s")
            print(f"     answer: {answer[:120]}")
            print()

        print(f"  Citation accuracy: {citation_hits}/{len(LLM_TEST_CASES)}")
        print(f"  Total tokens: {total_tokens}  |  Avg time: {total_time/len(LLM_TEST_CASES):.1f}s\n")

        llm_summary.append({
            "name":      llm_cfg["name"],
            "citations": f"{citation_hits}/{len(LLM_TEST_CASES)}",
            "tokens":    total_tokens,
            "avg_time":  round(total_time / len(LLM_TEST_CASES), 1),
        })


# ── Final summary ────────────────────────────────────────────────────────────
print(f"\n{BOLD_DIV}")
print("  RETRIEVAL SUMMARY")
print(f"{BOLD_DIV}")
print(f"  {'Config':<50} {'MRR':>6}  {'R@3':>6}  {'R@5':>6}  {'Misses':>7}")
print(f"  {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
best_mrr = max(r["mrr"] for r in retrieval_summary)
for r in retrieval_summary:
    marker = " ←" if r["mrr"] == best_mrr else ""
    print(f"  {r['name']:<50} {r['mrr']:>6.3f}  {r['rec@3']:>6.3f}  {r['rec@5']:>6.3f}  {r['misses']:>7}{marker}")

if llm_summary:
    print(f"\n{BOLD_DIV}")
    print("  LLM SUMMARY")
    print(f"{BOLD_DIV}")
    print(f"  {'Model':<25} {'Citations':>10}  {'Tokens':>8}  {'Avg time':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*10}")
    for r in llm_summary:
        print(f"  {r['name']:<25} {r['citations']:>10}  {r['tokens']:>8}  {r['avg_time']:>9.1f}s")

print(f"{BOLD_DIV}\n")
