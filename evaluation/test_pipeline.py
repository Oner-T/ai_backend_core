"""
KVKK RAG — Pipeline Smoke Test
================================
Runs the golden dataset through the pipeline and prints results for manual review.
No external API calls, no token cost.

Usage:
    source venv/bin/activate
    python evaluation/test_pipeline.py

    # Run only a specific scenario:
    python evaluation/test_pipeline.py cross_reference
    python evaluation/test_pipeline.py out_of_scope
    python evaluation/test_pipeline.py specific_madde
"""

import json
import os
import sys
import django

# ── Bootstrap Django ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_api.settings")
django.setup()

from intelligence.services import query_kvkk  # noqa

DIVIDER     = "─" * 70
BOLD_DIV    = "═" * 70

# ── Load dataset ─────────────────────────────────────────────────────────────
dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
with open(dataset_path) as f:
    golden = json.load(f)

# Optional: filter by scenario passed as CLI arg
scenario_filter = sys.argv[1] if len(sys.argv) > 1 else None
if scenario_filter:
    golden = [e for e in golden if e["scenario"] == scenario_filter]
    if not golden:
        print(f"❌  No entries found for scenario '{scenario_filter}'")
        sys.exit(1)

total = len(golden)
print(f"\n{BOLD_DIV}")
print(f"  KVKK Pipeline Test  —  {total} questions"
      + (f"  [filter: {scenario_filter}]" if scenario_filter else ""))
print(f"{BOLD_DIV}\n")

passed = 0
failed = 0
results = []

for i, entry in enumerate(golden):
    question        = entry["question"]
    expected_answer = entry["expected_answer"]
    expected_maddes = set(entry["expected_maddes"])
    scenario        = entry["scenario"]

    print(f"[{i+1:02d}/{total}] {scenario.upper()}")
    print(f"Q: {question}")

    result = query_kvkk(question, history=[])

    retrieved_maddes = {s["madde"] for s in result["sources"] if s["madde"]}

    # ── Retrieval check ───────────────────────────────────────────────────────
    if scenario == "out_of_scope":
        rejected = "Bu soru KVKK" in result["answer"] or len(result["sources"]) == 0
        status = "✅ PASS" if rejected else "❌ FAIL"
        if rejected:
            passed += 1
        else:
            failed += 1
        print(f"Rejection: {status}")
        if not rejected:
            print(f"⚠️  Expected rejection but got answer: {result['answer'][:150]}")
    else:
        if expected_maddes:
            correct   = expected_maddes & retrieved_maddes
            missing   = expected_maddes - retrieved_maddes
            extra     = retrieved_maddes - expected_maddes
            precision = len(correct) / len(retrieved_maddes) if retrieved_maddes else 0
            recall    = len(correct) / len(expected_maddes)
        else:
            missing = extra = set()
            precision = recall = 1.0

        retrieval_ok = recall == 1.0
        status = "✅ PASS" if retrieval_ok else "❌ FAIL"
        if retrieval_ok:
            passed += 1
        else:
            failed += 1

        print(f"Retrieval: {status}  |  expected={sorted(expected_maddes)}  got={sorted(retrieved_maddes)}")
        if missing:
            print(f"  ⚠️  Missing maddes: {sorted(missing)}")
        if extra:
            print(f"  ℹ️  Extra maddes: {sorted(extra)}  (may be cross-refs)")

        # ── Answer preview ────────────────────────────────────────────────────
        print(f"\nExpected:  {expected_answer[:200] if expected_answer else '—'}")
        print(f"Generated: {result['answer'][:200]}")

    print(DIVIDER)
    results.append({
        "question": question,
        "scenario": scenario,
        "status":   status,
        "expected_maddes": sorted(expected_maddes),
        "retrieved_maddes": sorted(retrieved_maddes),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{BOLD_DIV}")
print(f"  SUMMARY")
print(f"{BOLD_DIV}")
print(f"  Total:  {total}")
print(f"  Passed: {passed}  ✅")
print(f"  Failed: {failed}  ❌")
print(f"  Score:  {passed/total:.0%}")

if failed > 0:
    print(f"\n  Failed questions:")
    for r in results:
        if "FAIL" in r["status"]:
            print(f"    ❌ [{r['scenario']}] {r['question'][:65]}")
            if r["scenario"] != "out_of_scope":
                print(f"       expected={r['expected_maddes']}  got={r['retrieved_maddes']}")

print(f"{BOLD_DIV}\n")
