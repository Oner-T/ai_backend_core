"""
KVKK RAG — Golden Dataset Evaluation
======================================
Usage:
    ANTHROPIC_API_KEY=sk-... python evaluation/evaluate.py

Requires:
    pip install ragas datasets langchain-anthropic
"""

import json
import os
import sys
import django

# ── Bootstrap Django ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_api.settings")
django.setup()

from intelligence.services import query_kvkk  # noqa: E402 (must come after django.setup)

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

# ── Judge model ─────────────────────────────────────────────────────────────
# Using Claude Haiku as the judge — different from our Gemini answering model.
# Set ANTHROPIC_API_KEY in your environment before running.
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("❌  ANTHROPIC_API_KEY not set. Export it before running:")
    print("    export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

judge_llm = LangchainLLMWrapper(
    ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
)

# ── Load golden dataset ─────────────────────────────────────────────────────
dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
with open(dataset_path) as f:
    golden = json.load(f)

# ── Run pipeline on each question ────────────────────────────────────────────
questions          = []
generated_answers  = []
retrieved_contexts = []
reference_answers  = []
retrieval_results  = []

total = len(golden)
print(f"\n🚀 Running pipeline on {total} questions...\n")

for i, entry in enumerate(golden):
    question        = entry["question"]
    expected_maddes = set(entry["expected_maddes"])
    scenario        = entry["scenario"]

    print(f"  [{i+1:02d}/{total}] {question[:70]}")

    result = query_kvkk(question, history=[])

    retrieved_maddes = {s["madde"] for s in result["sources"] if s["madde"]}

    # ── Out-of-scope: just check rejection ──────────────────────────────────
    if scenario == "out_of_scope":
        rejected = "Bu soru KVKK" in result["answer"] or len(result["sources"]) == 0
        retrieval_results.append({
            "question": question,
            "scenario": scenario,
            "out_of_scope_correctly_rejected": rejected,
            "answer_preview": result["answer"][:120],
        })
        continue

    # ── Retrieval precision / recall ─────────────────────────────────────────
    if expected_maddes:
        correct = expected_maddes & retrieved_maddes
        precision = len(correct) / len(retrieved_maddes) if retrieved_maddes else 0.0
        recall    = len(correct) / len(expected_maddes)
    else:
        precision = recall = 1.0

    retrieval_results.append({
        "question":        question,
        "scenario":        scenario,
        "expected_maddes": sorted(expected_maddes),
        "retrieved_maddes": sorted(retrieved_maddes),
        "precision":       round(precision, 2),
        "recall":          round(recall, 2),
    })

    # ── Collect for RAGAS ───────────────────────────────────────────────────
    questions.append(question)
    generated_answers.append(result["answer"])
    retrieved_contexts.append([s["content"] for s in result["sources"]])
    reference_answers.append(entry["expected_answer"])

print(f"\n✅ Pipeline done. {len(questions)} questions collected for RAGAS.\n")

# ── RAGAS evaluation ─────────────────────────────────────────────────────────
print("🧠 Running RAGAS evaluation (calling Claude Haiku as judge)...\n")

ragas_dataset = Dataset.from_dict({
    "question":     questions,
    "answer":       generated_answers,
    "contexts":     retrieved_contexts,
    "ground_truth": reference_answers,
})

ragas_result = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

# ── Print results ─────────────────────────────────────────────────────────────
DIVIDER = "=" * 65

print(f"\n{DIVIDER}")
print("  RAGAS SCORES")
print(DIVIDER)
print(f"  Faithfulness       {ragas_result['faithfulness']:.3f}   LLM stays within retrieved context")
print(f"  Answer Relevancy   {ragas_result['answer_relevancy']:.3f}   Answer addresses the question")
print(f"  Context Precision  {ragas_result['context_precision']:.3f}   Retrieved chunks are signal, not noise")
print(f"  Context Recall     {ragas_result['context_recall']:.3f}   Relevant chunks not missed")
print(DIVIDER)

print(f"\n{DIVIDER}")
print("  RETRIEVAL RESULTS PER QUESTION")
print(DIVIDER)

for r in retrieval_results:
    if r["scenario"] == "out_of_scope":
        icon = "✅" if r["out_of_scope_correctly_rejected"] else "❌"
        print(f"  {icon} [out_of_scope]  {r['question'][:60]}")
        if not r["out_of_scope_correctly_rejected"]:
            print(f"       ⚠️  Not rejected — answer: {r['answer_preview']}")
    else:
        p_icon = "✅" if r["precision"] == 1.0 else "⚠️ "
        r_icon = "✅" if r["recall"]    == 1.0 else "❌"
        print(
            f"  P:{r['precision']:.2f}{p_icon}  R:{r['recall']:.2f}{r_icon}  "
            f"[{r['scenario']}]"
        )
        print(
            f"       expected={r['expected_maddes']}  "
            f"got={r['retrieved_maddes']}"
        )
        print(f"       {r['question'][:70]}")

# ── Aggregates ────────────────────────────────────────────────────────────────
scored = [r for r in retrieval_results if "precision" in r]
out_of_scope_entries = [r for r in retrieval_results if r["scenario"] == "out_of_scope"]

avg_p = sum(r["precision"] for r in scored) / len(scored) if scored else 0
avg_r = sum(r["recall"]    for r in scored) / len(scored) if scored else 0
rejection_rate = (
    sum(1 for r in out_of_scope_entries if r["out_of_scope_correctly_rejected"])
    / len(out_of_scope_entries)
    if out_of_scope_entries else 0
)

print(f"\n{DIVIDER}")
print("  RETRIEVAL SUMMARY")
print(DIVIDER)
print(f"  Average Precision:       {avg_p:.3f}")
print(f"  Average Recall:          {avg_r:.3f}")
print(f"  Out-of-scope rejection:  {rejection_rate:.0%}  "
      f"({sum(1 for r in out_of_scope_entries if r['out_of_scope_correctly_rejected'])}"
      f"/{len(out_of_scope_entries)} correctly rejected)")
print(DIVIDER)

# ── Save results to JSON ──────────────────────────────────────────────────────
output = {
    "ragas": {
        "faithfulness":      ragas_result["faithfulness"],
        "answer_relevancy":  ragas_result["answer_relevancy"],
        "context_precision": ragas_result["context_precision"],
        "context_recall":    ragas_result["context_recall"],
    },
    "retrieval": {
        "avg_precision":          round(avg_p, 3),
        "avg_recall":             round(avg_r, 3),
        "out_of_scope_rejection": round(rejection_rate, 3),
    },
    "per_question": retrieval_results,
}

output_path = os.path.join(os.path.dirname(__file__), "results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n💾 Full results saved to evaluation/results.json\n")
