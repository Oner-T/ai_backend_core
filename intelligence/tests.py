"""
Unit tests for the intelligence app.

Three groups:
  1. API views  — QueryView and FeedbackView, query_kvkk mocked out
  2. Metric logic — retrieval recall, citation, nDCG, MRR (pure Python, no DB)
  3. Models — UserFeedback and GroundTruthEntry creation
"""

import math
import re
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from intelligence.models import GroundTruthEntry, UserFeedback


# ── Helpers (same logic as run_evaluation / run_langsmith_evaluation) ─────────

def _compute_retrieval(expected_sources, returned_sources):
    expected_pairs = {
        (s["document_name"], s["madde"])
        for s in expected_sources
        if s.get("document_name") and s.get("madde") is not None
    }
    returned_pairs = {
        (s["document_name"], s["madde"])
        for s in returned_sources
        if s.get("document_name") and s.get("madde") is not None
    }
    if expected_pairs:
        overlap    = expected_pairs & returned_pairs
        recall     = len(overlap) / len(expected_pairs)
        hit        = bool(overlap)
    else:
        hit    = len(returned_pairs) == 0
        recall = 1.0 if hit else 0.0
    return recall, hit


def _compute_citation(expected_sources, answer_text):
    expected_maddes = {s["madde"] for s in expected_sources if s.get("madde") is not None}
    cited = {int(m) for m in re.findall(r'Madde\s+(\d+)', answer_text, re.IGNORECASE)}
    return bool(expected_maddes & cited) if expected_maddes else True


def _ndcg(expected_pairs: set, returned_pairs: list, k: int = 8) -> float:
    if not expected_pairs:
        return 1.0
    gains = [1.0 if p in expected_pairs else 0.0 for p in returned_pairs[:k]]
    dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_pairs), k)))
    return dcg / ideal if ideal else 0.0


def _mrr(expected_pairs: set, returned_pairs: list) -> float:
    for i, p in enumerate(returned_pairs):
        if p in expected_pairs:
            return 1.0 / (i + 1)
    return 0.0


# ── Fake pipeline response ────────────────────────────────────────────────────

MOCK_PIPELINE_RESPONSE = {
    "answer": "KVKK Madde 3'e göre kişisel veri, kimliği belirli kişiye ilişkin bilgidir.",
    "model":  "gemini-2.5-flash",
    "sources": [
        {"document_name": "KVKK", "madde": 3, "bolum": 1, "madde_title": "Tanımlar", "content": "..."},
        {"document_name": "KVKK", "madde": 4, "bolum": 2, "madde_title": "Genel ilkeler", "content": "..."},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. API VIEW TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class QueryViewTests(TestCase):

    def setUp(self):
        self.client = APIClient()

    @patch("intelligence.views.query_kvkk", return_value=MOCK_PIPELINE_RESPONSE)
    def test_valid_question_returns_200(self, _mock):
        res = self.client.post("/api/query/", {"question": "Kişisel veri nedir?"}, format="json")
        self.assertEqual(res.status_code, 200)

    @patch("intelligence.views.query_kvkk", return_value=MOCK_PIPELINE_RESPONSE)
    def test_response_contains_answer_model_sources(self, _mock):
        res = self.client.post("/api/query/", {"question": "Kişisel veri nedir?"}, format="json")
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("model", data)
        self.assertIn("sources", data)

    @patch("intelligence.views.query_kvkk", return_value=MOCK_PIPELINE_RESPONSE)
    def test_history_is_passed_to_pipeline(self, mock_query):
        history = [{"role": "user", "content": "Merhaba"}]
        self.client.post("/api/query/", {"question": "Devam?", "history": history}, format="json")
        mock_query.assert_called_once_with("Devam?", history=history)

    def test_missing_question_returns_400(self):
        res = self.client.post("/api/query/", {}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_empty_question_returns_400(self):
        res = self.client.post("/api/query/", {"question": "   "}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_get_method_not_allowed(self):
        res = self.client.get("/api/query/")
        self.assertEqual(res.status_code, 405)


class FeedbackViewTests(TestCase):

    def setUp(self):
        self.client  = APIClient()
        self.payload = {
            "question": "Kişisel veri nedir?",
            "answer":   "Gerçek kişiye ilişkin her türlü bilgi.",
            "rating":   "good",
            "sources":  [{"document_name": "KVKK", "madde": 3}],
            "comment":  "",
        }

    def test_valid_good_feedback_returns_201(self):
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 201)

    def test_valid_bad_feedback_returns_201(self):
        self.payload["rating"] = "bad"
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 201)

    def test_feedback_saved_to_db(self):
        self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(UserFeedback.objects.count(), 1)
        fb = UserFeedback.objects.first()
        self.assertEqual(fb.question, "Kişisel veri nedir?")
        self.assertEqual(fb.rating, "good")

    def test_missing_question_returns_400(self):
        del self.payload["question"]
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 400)

    def test_missing_answer_returns_400(self):
        del self.payload["answer"]
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 400)

    def test_invalid_rating_returns_400(self):
        self.payload["rating"] = "meh"
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 400)

    def test_missing_rating_returns_400(self):
        del self.payload["rating"]
        res = self.client.post("/api/feedback/", self.payload, format="json")
        self.assertEqual(res.status_code, 400)

    def test_optional_comment_defaults_to_empty(self):
        del self.payload["comment"]
        self.client.post("/api/feedback/", self.payload, format="json")
        fb = UserFeedback.objects.first()
        self.assertEqual(fb.comment, "")

    def test_optional_sources_defaults_to_empty_list(self):
        del self.payload["sources"]
        self.client.post("/api/feedback/", self.payload, format="json")
        fb = UserFeedback.objects.first()
        self.assertEqual(fb.sources, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION METRIC LOGIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalMetricTests(TestCase):

    def test_full_overlap_gives_100_recall(self):
        expected = [{"document_name": "KVKK", "madde": 3}]
        returned = [{"document_name": "KVKK", "madde": 3}]
        recall, hit = _compute_retrieval(expected, returned)
        self.assertEqual(recall, 1.0)
        self.assertTrue(hit)

    def test_partial_overlap_gives_partial_recall(self):
        expected = [
            {"document_name": "KVKK", "madde": 3},
            {"document_name": "KVKK", "madde": 6},
        ]
        returned = [{"document_name": "KVKK", "madde": 3}]
        recall, hit = _compute_retrieval(expected, returned)
        self.assertAlmostEqual(recall, 0.5)
        self.assertTrue(hit)

    def test_no_overlap_gives_zero_recall(self):
        expected = [{"document_name": "KVKK", "madde": 6}]
        returned = [{"document_name": "KVKK", "madde": 3}]
        recall, hit = _compute_retrieval(expected, returned)
        self.assertEqual(recall, 0.0)
        self.assertFalse(hit)

    def test_out_of_scope_passes_when_no_sources_returned(self):
        recall, hit = _compute_retrieval([], [])
        self.assertEqual(recall, 1.0)
        self.assertTrue(hit)

    def test_out_of_scope_fails_when_sources_returned(self):
        returned = [{"document_name": "KVKK", "madde": 3}]
        recall, hit = _compute_retrieval([], returned)
        self.assertEqual(recall, 0.0)
        self.assertFalse(hit)

    def test_extra_returned_sources_dont_hurt_recall(self):
        expected = [{"document_name": "KVKK", "madde": 3}]
        returned = [
            {"document_name": "KVKK", "madde": 3},
            {"document_name": "KVKK", "madde": 12},
            {"document_name": "KVKK", "madde": 28},
        ]
        recall, hit = _compute_retrieval(expected, returned)
        self.assertEqual(recall, 1.0)
        self.assertTrue(hit)


class CitationMetricTests(TestCase):

    def test_exact_madde_citation_passes(self):
        expected = [{"document_name": "KVKK", "madde": 3}]
        answer   = "KVKK Madde 3'e göre kişisel veri tanımı şöyledir."
        self.assertTrue(_compute_citation(expected, answer))

    def test_citation_is_case_insensitive(self):
        expected = [{"document_name": "KVKK", "madde": 5}]
        answer   = "madde 5 uyarınca açık rıza gerekir."
        self.assertTrue(_compute_citation(expected, answer))

    def test_wrong_madde_citation_fails(self):
        expected = [{"document_name": "KVKK", "madde": 6}]
        answer   = "KVKK Madde 3'e göre kişisel veri tanımı şöyledir."
        self.assertFalse(_compute_citation(expected, answer))

    def test_no_citation_fails(self):
        expected = [{"document_name": "KVKK", "madde": 3}]
        answer   = "Bu konuda bilgi bulunmamaktadır."
        self.assertFalse(_compute_citation(expected, answer))

    def test_out_of_scope_citation_always_passes(self):
        self.assertTrue(_compute_citation([], "Herhangi bir metin."))

    def test_multiple_expected_any_match_passes(self):
        expected = [
            {"document_name": "KVKK", "madde": 9},
            {"document_name": "Aktarım Yönetmeliği", "madde": 4},
        ]
        answer = "KVKK Madde 9 kapsamında yurt dışı aktarım incelenir."
        self.assertTrue(_compute_citation(expected, answer))


class NDCGTests(TestCase):

    def test_perfect_ranking_gives_1(self):
        expected = {("KVKK", 3)}
        returned = [("KVKK", 3), ("KVKK", 5), ("KVKK", 12)]
        self.assertAlmostEqual(_ndcg(expected, returned), 1.0)

    def test_correct_source_at_rank2_less_than_rank1(self):
        expected = {("KVKK", 3)}
        at_rank1 = [("KVKK", 3), ("KVKK", 5)]
        at_rank2 = [("KVKK", 5), ("KVKK", 3)]
        self.assertGreater(_ndcg(expected, at_rank1), _ndcg(expected, at_rank2))

    def test_correct_source_not_found_gives_0(self):
        expected = {("KVKK", 6)}
        returned = [("KVKK", 3), ("KVKK", 5)]
        self.assertAlmostEqual(_ndcg(expected, returned), 0.0)

    def test_empty_expected_gives_1(self):
        self.assertAlmostEqual(_ndcg(set(), [("KVKK", 3)]), 1.0)


class MRRTests(TestCase):

    def test_first_rank_gives_1(self):
        expected = {("KVKK", 3)}
        returned = [("KVKK", 3), ("KVKK", 5)]
        self.assertAlmostEqual(_mrr(expected, returned), 1.0)

    def test_second_rank_gives_half(self):
        expected = {("KVKK", 3)}
        returned = [("KVKK", 5), ("KVKK", 3)]
        self.assertAlmostEqual(_mrr(expected, returned), 0.5)

    def test_not_found_gives_0(self):
        expected = {("KVKK", 6)}
        returned = [("KVKK", 3), ("KVKK", 5)]
        self.assertAlmostEqual(_mrr(expected, returned), 0.0)

    def test_fifth_rank_gives_point_two(self):
        expected = {("KVKK", 6)}
        returned = [("KVKK", 1), ("KVKK", 2), ("KVKK", 3), ("KVKK", 4), ("KVKK", 6)]
        self.assertAlmostEqual(_mrr(expected, returned), 0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class UserFeedbackModelTests(TestCase):

    def _make_feedback(self, **kwargs):
        defaults = dict(
            question="Kişisel veri nedir?",
            answer="Gerçek kişiye ilişkin her türlü bilgi.",
            sources=[{"document_name": "KVKK", "madde": 3}],
            rating="good",
        )
        defaults.update(kwargs)
        return UserFeedback.objects.create(**defaults)

    def test_feedback_created_successfully(self):
        fb = self._make_feedback()
        self.assertEqual(UserFeedback.objects.count(), 1)
        self.assertEqual(fb.rating, "good")

    def test_str_representation(self):
        fb = self._make_feedback()
        self.assertIn("good", str(fb))
        self.assertIn("Kişisel veri", str(fb))

    def test_comment_defaults_to_empty_string(self):
        fb = self._make_feedback()
        self.assertEqual(fb.comment, "")

    def test_is_not_promoted_by_default(self):
        fb = self._make_feedback()
        self.assertFalse(fb.ground_truth.exists())


class GroundTruthEntryModelTests(TestCase):

    def _make_entry(self, **kwargs):
        defaults = dict(
            question="Kişisel veri nedir?",
            expected_answer="Gerçek kişiye ilişkin her türlü bilgi.",
            expected_sources=[{"document_name": "KVKK", "madde": 3}],
            scenario="definition",
        )
        defaults.update(kwargs)
        return GroundTruthEntry.objects.create(**defaults)

    def test_entry_created_successfully(self):
        entry = self._make_entry()
        self.assertEqual(GroundTruthEntry.objects.count(), 1)
        self.assertEqual(entry.scenario, "definition")

    def test_str_representation(self):
        entry = self._make_entry()
        self.assertIn("Kişisel veri", str(entry))

    def test_question_is_unique(self):
        self._make_entry()
        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            self._make_entry()

    def test_promoted_from_links_feedback(self):
        fb = UserFeedback.objects.create(
            question="Soru",
            answer="Cevap",
            sources=[],
            rating="good",
        )
        entry = self._make_entry(question="Soru promovecek mi?", promoted_from=fb)
        self.assertEqual(entry.promoted_from, fb)
        self.assertTrue(fb.ground_truth.exists())

    def test_expected_sources_stored_as_list(self):
        entry = self._make_entry()
        self.assertIsInstance(entry.expected_sources, list)
        self.assertEqual(entry.expected_sources[0]["madde"], 3)
