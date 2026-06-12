"""
SVD-based Recommender Model
============================

Wraps scikit-surprise matrix factorization models (SVD, SVD++, NMF)
for the Netflix Prize recommendation system. Provides a unified interface
for training, prediction, hyperparameter tuning, and serialization.

Example
-------
>>> from src.models.svd_model import SVDRecommender
>>> model = SVDRecommender(model_type='svd', n_factors=100, n_epochs=20)
>>> model.fit(train_df)
>>> pred = model.predict(user_id=1, movie_id=42)
>>> top_k = model.recommend_top_k(user_id=1, all_movie_ids=movie_ids, k=10)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from surprise import (
    Dataset,
    NMF,
    SVD,
    SVDpp,
    Reader,
    accuracy,
)
from surprise.model_selection import GridSearchCV as SurpriseGridSearchCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default hyper-parameter grids for grid search
# ---------------------------------------------------------------------------
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "svd": {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30],
        "lr_all": [0.002, 0.005, 0.01],
        "reg_all": [0.02, 0.05, 0.1],
    },
    "svdpp": {
        "n_factors": [20, 50],
        "n_epochs": [20, 30],
        "lr_all": [0.005, 0.01],
        "reg_all": [0.02, 0.1],
    },
    "nmf": {
        "n_factors": [15, 50, 100],
        "n_epochs": [30, 50],
        "reg_pu": [0.06, 0.1],
        "reg_qi": [0.06, 0.1],
    },
}

# Map friendly names → surprise algorithm classes
_ALGO_MAP = {
    "svd": SVD,
    "svdpp": SVDpp,
    "nmf": NMF,
}


class SVDRecommender:
    """Surprise-based matrix factorisation recommender.

    Parameters
    ----------
    model_type : str
        One of ``'svd'``, ``'svdpp'``, ``'nmf'``.
    **kwargs
        Forwarded directly to the underlying surprise algorithm constructor
        (e.g. ``n_factors``, ``n_epochs``, ``lr_all``, ``reg_all``).

    Attributes
    ----------
    algo : surprise.AlgoBase
        The fitted surprise algorithm instance.
    model_type : str
        String identifier for the algorithm family.
    trainset : surprise.Trainset | None
        The internal surprise trainset created from the last ``fit`` call.
    """

    RATING_SCALE: Tuple[float, float] = (1.0, 5.0)

    def __init__(self, model_type: str = "svd", **kwargs: Any) -> None:
        model_type = model_type.lower().strip()
        if model_type not in _ALGO_MAP:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from {list(_ALGO_MAP.keys())}."
            )
        self.model_type: str = model_type
        self._algo_kwargs: Dict[str, Any] = kwargs
        self.algo = _ALGO_MAP[model_type](**kwargs)
        self.trainset = None
        logger.info(
            "Initialised %s recommender with params: %s",
            model_type.upper(),
            kwargs or "defaults",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _df_to_surprise_dataset(
        df: pd.DataFrame,
        rating_scale: Tuple[float, float] = (1.0, 5.0),
    ) -> Dataset:
        """Convert a DataFrame with [user_id, movie_id, rating] to a
        :class:`surprise.Dataset`.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns ``user_id``, ``movie_id``, ``rating``.
        rating_scale : tuple of float
            ``(min_rating, max_rating)`` for the ``Reader``.

        Returns
        -------
        surprise.Dataset
        """
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        reader = Reader(rating_scale=rating_scale)
        return Dataset.load_from_df(
            df[["user_id", "movie_id", "rating"]], reader
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        """Train the model on the provided DataFrame.

        Parameters
        ----------
        train_df : pd.DataFrame
            Columns: ``user_id``, ``movie_id``, ``rating``.

        Returns
        -------
        SVDRecommender
            ``self`` (for method chaining).
        """
        logger.info(
            "Fitting %s on %d ratings …", self.model_type.upper(), len(train_df)
        )
        dataset = self._df_to_surprise_dataset(
            train_df, rating_scale=self.RATING_SCALE
        )
        self.trainset = dataset.build_full_trainset()
        self.algo.fit(self.trainset)
        logger.info("Fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, user_id: int, movie_id: int) -> float:
        """Return a single predicted rating.

        Parameters
        ----------
        user_id : int
            The raw user id.
        movie_id : int
            The raw movie (item) id.

        Returns
        -------
        float
            Predicted rating clipped to ``RATING_SCALE``.
        """
        if self.trainset is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        pred = self.algo.predict(uid=user_id, iid=movie_id)
        return float(
            np.clip(pred.est, self.RATING_SCALE[0], self.RATING_SCALE[1])
        )

    def predict_batch(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for many (user, item) pairs.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns.

        Returns
        -------
        np.ndarray
            Array of predicted ratings, same length as ``pairs_df``.
        """
        if self.trainset is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        predictions = np.array(
            [
                self.predict(row.user_id, row.movie_id)
                for row in pairs_df.itertuples(index=False)
            ],
            dtype=np.float32,
        )
        return predictions

    # ------------------------------------------------------------------
    # Top-K recommendations
    # ------------------------------------------------------------------
    def recommend_top_k(
        self,
        user_id: int,
        all_movie_ids: Union[List[int], np.ndarray],
        k: int = 10,
        exclude_seen: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return top-*k* movie recommendations for a user.

        Parameters
        ----------
        user_id : int
            The target user.
        all_movie_ids : list or array of int
            Universe of candidate movie ids.
        k : int
            Number of recommendations to return.
        exclude_seen : set of int or None
            Movie ids to exclude (e.g. movies the user has already rated).

        Returns
        -------
        list of (movie_id, predicted_rating)
            Sorted descending by predicted rating.
        """
        if self.trainset is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        exclude_seen = exclude_seen or set()
        candidates = [mid for mid in all_movie_ids if mid not in exclude_seen]

        preds = [
            (mid, self.predict(user_id, mid))
            for mid in candidates
        ]
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:k]

    # ------------------------------------------------------------------
    # Hyper-parameter tuning
    # ------------------------------------------------------------------
    def grid_search(
        self,
        train_df: pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        measures: Optional[List[str]] = None,
        cv: int = 3,
        n_jobs: int = -1,
        refit: bool = True,
    ) -> Dict[str, Any]:
        """Run grid search cross-validation and (optionally) refit on full
        training data with the best parameters.

        Parameters
        ----------
        train_df : pd.DataFrame
            Columns: ``user_id``, ``movie_id``, ``rating``.
        param_grid : dict or None
            Parameter grid. If *None*, a sensible default grid for the
            current ``model_type`` is used.
        measures : list of str or None
            Evaluation measures (default ``['rmse', 'mae']``).
        cv : int
            Number of cross-validation folds.
        n_jobs : int
            Number of parallel jobs (``-1`` = all CPUs).
        refit : bool
            If *True*, refit the model with the best parameters on the
            full training set.

        Returns
        -------
        dict
            ``{'best_params': dict, 'best_score': float, 'cv_results': dict}``
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRIDS.get(self.model_type, {})
            logger.info(
                "Using default param grid for %s: %s",
                self.model_type,
                param_grid,
            )

        measures = measures or ["rmse", "mae"]
        dataset = self._df_to_surprise_dataset(
            train_df, rating_scale=self.RATING_SCALE
        )

        algo_cls = _ALGO_MAP[self.model_type]
        gs = SurpriseGridSearchCV(
            algo_cls,
            param_grid,
            measures=measures,
            cv=cv,
            n_jobs=n_jobs,
            refit=False,  # we handle refit ourselves to store trainset
        )
        gs.fit(dataset)

        best_params: Dict[str, Any] = gs.best_params["rmse"]
        best_score: float = gs.best_score["rmse"]
        logger.info("Best RMSE: %.4f | Best params: %s", best_score, best_params)

        if refit:
            self.algo = algo_cls(**best_params)
            self._algo_kwargs = best_params
            self.fit(train_df)
            logger.info("Refitted model with best params on full training set.")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": gs.cv_results,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Persist the model to disk via pickle.

        Parameters
        ----------
        path : str or Path
            File path (e.g. ``'models/svd.pkl'``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_type": self.model_type,
            "algo_kwargs": self._algo_kwargs,
            "algo": self.algo,
            "trainset": self.trainset,
            "rating_scale": self.RATING_SCALE,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SVDRecommender":
        """Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file.

        Returns
        -------
        SVDRecommender
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls.__new__(cls)
        instance.model_type = state["model_type"]
        instance._algo_kwargs = state["algo_kwargs"]
        instance.algo = state["algo"]
        instance.trainset = state["trainset"]
        instance.RATING_SCALE = state.get("rating_scale", (1.0, 5.0))
        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"SVDRecommender(model_type='{self.model_type}', "
            f"params={self._algo_kwargs})"
        )
