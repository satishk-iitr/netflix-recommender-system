"""
Unit tests for evaluation metrics.
Run with: pytest tests/ -v
"""
import math
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    rmse, mae, precision_at_k, recall_at_k,
    ap_at_k, map_at_k, ndcg_at_k, coverage, hit_rate
)


class TestRMSE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([3.0, 4.0, 5.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # each off by 1
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            rmse(np.array([1.0, 2.0]), np.array([1.0]))


class TestMAE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)


class TestPrecisionAtK:
    def test_all_relevant(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recommended, relevant, k=5) == pytest.approx(1.0)

    def test_none_relevant(self):
        recommended = [1, 2, 3]
        relevant = {10, 11, 12}
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(0.0)

    def test_half_relevant(self):
        recommended = [1, 2, 3, 4]
        relevant = {1, 3}  # 2 out of 4
        assert precision_at_k(recommended, relevant, k=4) == pytest.approx(0.5)

    def test_k_truncation(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {4, 5}  # only in positions 4,5 — beyond k=3
        assert precision_at_k(recommended, relevant, k=3) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_found(self):
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert recall_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_empty_relevant(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)

    def test_partial(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 10, 11}  # 2 found out of 4
        assert recall_at_k(recommended, relevant, k=5) == pytest.approx(0.5)


class TestAPAtK:
    def test_perfect_ranking(self):
        # All top-k items relevant → AP = 1.0
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert ap_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ap_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)

    def test_single_hit_at_rank_1(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1}  # one relevant item at rank 1
        # AP = (1/1) * P(1) * rel(1) = (1/1) * (1/1) * 1 = 1.0
        assert ap_at_k(recommended, relevant, k=5) == pytest.approx(1.0)

    def test_single_hit_at_rank_2(self):
        recommended = [9, 1, 3, 4, 5]
        relevant = {1}  # relevant item at rank 2
        # AP = (1/min(1,5)) * P(2) * rel(2) = 1 * (1/2) * 1 = 0.5
        assert ap_at_k(recommended, relevant, k=5) == pytest.approx(0.5)

    def test_relevance_threshold_is_external(self):
        """AP@K receives pre-filtered relevant sets — threshold applied upstream."""
        recommended = [1, 2, 3]
        relevant = {1, 2}  # only items rated >= 3.5 should be in this set
        result = ap_at_k(recommended, relevant, k=3)
        assert 0.0 <= result <= 1.0


class TestMAPAtK:
    def test_single_user_perfect(self):
        user_recs = {1: [1, 2, 3]}
        user_relevant = {1: {1, 2, 3}}
        assert map_at_k(user_recs, user_relevant, k=3) == pytest.approx(1.0)

    def test_excludes_users_no_relevant(self):
        """Users with 0 relevant items must NOT contribute to MAP."""
        user_recs = {1: [1, 2, 3], 2: [4, 5, 6]}
        user_relevant = {1: {1, 2, 3}, 2: set()}  # user 2 has no relevant
        result = map_at_k(user_recs, user_relevant, k=3)
        # Only user 1 contributes → AP@3 for user 1 = 1.0
        assert result == pytest.approx(1.0)

    def test_empty_recommendations(self):
        assert map_at_k({}, {}, k=10) == pytest.approx(0.0)

    def test_relevance_threshold_documented(self):
        """The default relevance_threshold param is 3.5 — verify it exists."""
        import inspect
        sig = inspect.signature(map_at_k)
        assert "relevance_threshold" in sig.parameters
        assert sig.parameters["relevance_threshold"].default == 3.5


class TestNDCGAtK:
    def test_perfect(self):
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert ndcg_at_k(recommended, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)

    def test_reversed_order_worse(self):
        # [1,2,3] all relevant — perfect DCG
        # [3,2,1] all relevant — same items, same DCG (binary relevance)
        recommended_a = [1, 2, 3]
        recommended_b = [3, 2, 1]
        relevant = {1, 2, 3}
        score_a = ndcg_at_k(recommended_a, relevant, k=3)
        score_b = ndcg_at_k(recommended_b, relevant, k=3)
        # Binary relevance: same positions same items → equal
        assert score_a == pytest.approx(score_b)


class TestCoverage:
    def test_full_coverage(self):
        recs = {1: [1, 2, 3], 2: [4, 5, 6]}
        assert coverage(recs, n_total_items=6) == pytest.approx(1.0)

    def test_partial_coverage(self):
        recs = {1: [1, 2, 3]}
        assert coverage(recs, n_total_items=6) == pytest.approx(0.5)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            coverage({1: [1]}, n_total_items=0)


class TestHitRate:
    def test_all_hit(self):
        user_recs = {1: [1, 2, 3], 2: [4, 5, 6]}
        user_relevant = {1: {1}, 2: {4}}
        assert hit_rate(user_recs, user_relevant) == pytest.approx(1.0)

    def test_no_hit(self):
        user_recs = {1: [1, 2, 3]}
        user_relevant = {1: {9, 10}}
        assert hit_rate(user_recs, user_relevant) == pytest.approx(0.0)

    def test_excludes_no_relevant(self):
        user_recs = {1: [1, 2], 2: [3, 4]}
        user_relevant = {1: {1}, 2: set()}  # user 2 excluded
        assert hit_rate(user_recs, user_relevant) == pytest.approx(1.0)
