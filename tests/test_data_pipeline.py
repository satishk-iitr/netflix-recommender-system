"""
Unit tests for data pipeline utilities.
Run with: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            "movie_id": [10, 20, 30, 10, 40, 10, 20, 30, 40, 50],
            "rating": [5, 4, 3, 2, 4, 5, 5, 4, 3, 5],
            "date": pd.to_datetime(["2002-01-01", "2002-02-01", "2002-03-01",
                                    "2003-01-01", "2003-06-01",
                                    "2001-01-01", "2001-06-01", "2002-01-01",
                                    "2003-01-01", "2004-01-01"]),
        })

    def test_filter_cold_start(self, sample_df):
        from src.data.preprocessor import DataPreprocessor
        dp = DataPreprocessor()
        filtered = dp.filter_cold_start(sample_df, min_user_ratings=3, min_movie_ratings=1)
        # user 1 has 3 ratings (exactly min), user 2 has 2 (excluded)
        assert 2 not in filtered["user_id"].values
        assert len(filtered) < len(sample_df)

    def test_encode_ids_contiguous(self, sample_df):
        from src.data.preprocessor import DataPreprocessor
        dp = DataPreprocessor()
        encoded = dp.encode_ids(sample_df)
        user_indices = sorted(encoded["user_idx"].unique())
        movie_indices = sorted(encoded["movie_idx"].unique())
        # Must be contiguous starting from 0
        assert user_indices == list(range(len(user_indices)))
        assert movie_indices == list(range(len(movie_indices)))

    def test_get_sparsity(self, sample_df):
        from src.data.preprocessor import DataPreprocessor
        dp = DataPreprocessor()
        sparsity = dp.get_sparsity(sample_df)
        n_users = sample_df["user_id"].nunique()
        n_items = sample_df["movie_id"].nunique()
        n_interactions = len(sample_df)
        expected = 1.0 - n_interactions / (n_users * n_items)
        assert sparsity == pytest.approx(expected)

    def test_temporal_split_respects_order(self, sample_df):
        """Train ratings must come BEFORE test ratings for each user."""
        from src.data.preprocessor import DataPreprocessor
        dp = DataPreprocessor()
        # Use a large enough dataset (user 3 has 5 ratings)
        df_u3 = sample_df[sample_df["user_id"] == 3].copy()
        train, val, test = dp.temporal_split(sample_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        if len(train) > 0 and len(test) > 0:
            train_u3 = train[train["user_id"] == 3]
            test_u3 = test[test["user_id"] == 3]
            if len(train_u3) > 0 and len(test_u3) > 0:
                assert train_u3["date"].max() <= test_u3["date"].min()

    def test_train_val_test_disjoint(self, sample_df):
        """No (user, item) pair should appear in multiple splits."""
        from src.data.preprocessor import DataPreprocessor
        dp = DataPreprocessor()
        train, val, test = dp.temporal_split(sample_df)
        train_pairs = set(zip(train["user_id"], train["movie_id"]))
        val_pairs = set(zip(val["user_id"], val["movie_id"]))
        test_pairs = set(zip(test["user_id"], test["movie_id"]))
        assert len(train_pairs & test_pairs) == 0
        assert len(train_pairs & val_pairs) == 0


class TestRatingDataset:
    """Tests for PyTorch datasets."""

    def test_rating_dataset_length(self):
        from src.data.dataset import RatingDataset
        import torch
        users = np.array([0, 1, 2])
        items = np.array([0, 1, 2])
        ratings = np.array([5.0, 4.0, 3.0])
        ds = RatingDataset(users, items, ratings)
        assert len(ds) == 3

    def test_rating_dataset_returns_tensors(self):
        from src.data.dataset import RatingDataset
        import torch
        users = np.array([0, 1])
        items = np.array([0, 1])
        ratings = np.array([5.0, 4.0])
        ds = RatingDataset(users, items, ratings)
        u, i, r = ds[0]
        assert isinstance(u, torch.Tensor)
        assert isinstance(i, torch.Tensor)
        assert isinstance(r, torch.Tensor)


class TestInteractionMatrix:
    """Tests for the sparse interaction matrix."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "user_idx": [0, 0, 1, 1, 2],
            "movie_idx": [0, 1, 0, 2, 1],
            "rating": [5, 4, 3, 5, 4],
        })

    def test_shape(self, sample_df):
        from src.data.dataset import InteractionMatrix
        n_users, n_items = 3, 3
        mat = InteractionMatrix(sample_df, n_users=n_users, n_items=n_items)
        assert mat.matrix.shape == (n_users, n_items)

    def test_get_user_items(self, sample_df):
        from src.data.dataset import InteractionMatrix
        mat = InteractionMatrix(sample_df, n_users=3, n_items=3)
        items_for_user_0 = mat.get_user_items(0)
        assert set(items_for_user_0) == {0, 1}

    def test_to_edge_index(self, sample_df):
        import torch
        from src.data.dataset import InteractionMatrix
        mat = InteractionMatrix(sample_df, n_users=3, n_items=3)
        ei = mat.to_edge_index()
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long
