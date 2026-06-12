"""
Ensemble Methods for Netflix Recommendation
=============================================

Provides two ensemble strategies for combining predictions from multiple
recommendation models:

1. **WeightedEnsemble** — weighted average of base model predictions with
   optional weight optimisation via scipy.
2. **StackingEnsemble** — trains a LightGBM meta-learner on base model
   predictions + hand-crafted features (user/item statistics).

Example
-------
>>> from src.models.ensemble import WeightedEnsemble, StackingEnsemble
>>> we = WeightedEnsemble(models={'svd': svd_model, 'neumf': neumf_trainer})
>>> we.optimize_weights(val_df)
>>> we.predict(user_id=1, movie_id=42)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ======================================================================
# Protocol for base models (structural subtyping)
# ======================================================================
class _BaseModel(Protocol):
    """Minimal interface expected from any base model."""

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...
    def predict_batch(self, pairs_df: pd.DataFrame) -> np.ndarray: ...


# ======================================================================
# Weighted Ensemble
# ======================================================================
class WeightedEnsemble:
    """Weighted average of multiple recommendation models.

    Parameters
    ----------
    models : dict[str, model]
        Mapping of model names to model objects.  Each model must expose
        a ``predict_batch(pairs_df)`` method returning an ndarray.
    weights : dict[str, float] or None
        Initial weights.  If *None*, uniform weights are used.  Weights
        are automatically normalised to sum to 1.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if not models:
            raise ValueError("Must provide at least one model.")
        self.models = models
        self.model_names: List[str] = sorted(models.keys())

        if weights is None:
            n = len(self.model_names)
            self.weights: Dict[str, float] = {
                name: 1.0 / n for name in self.model_names
            }
        else:
            self.weights = self._normalise(weights)

        logger.info(
            "WeightedEnsemble initialised with models: %s, weights: %s",
            self.model_names,
            self.weights,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(w: Dict[str, float]) -> Dict[str, float]:
        total = sum(w.values())
        if total == 0:
            raise ValueError("Weights sum to zero.")
        return {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------
    def _get_all_predictions(
        self, pairs_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Collect predictions from every base model."""
        preds: Dict[str, np.ndarray] = {}
        for name in self.model_names:
            preds[name] = np.asarray(
                self.models[name].predict_batch(pairs_df), dtype=np.float64
            )
        return preds

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------
    def optimize_weights(
        self,
        val_df: pd.DataFrame,
        metric: str = "rmse",
    ) -> Dict[str, float]:
        """Find optimal model weights on a validation set.

        Uses ``scipy.optimize.minimize`` (SLSQP) with a simplex
        constraint (weights >= 0, sum = 1).

        Parameters
        ----------
        val_df : pd.DataFrame
            Must contain ``user_id``, ``movie_id``, ``rating``.
        metric : str
            ``'rmse'`` or ``'mae'``.

        Returns
        -------
        dict[str, float]
            Optimised weights (also stored in ``self.weights``).
        """
        true_ratings = val_df["rating"].values.astype(np.float64)
        preds = self._get_all_predictions(val_df)
        pred_matrix = np.column_stack(
            [preds[name] for name in self.model_names]
        )  # (N, M)

        def _objective(w: np.ndarray) -> float:
            blended = pred_matrix @ w
            if metric == "rmse":
                return float(np.sqrt(np.mean((blended - true_ratings) ** 2)))
            elif metric == "mae":
                return float(np.mean(np.abs(blended - true_ratings)))
            else:
                raise ValueError(f"Unknown metric '{metric}'")

        n = len(self.model_names)
        x0 = np.ones(n) / n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            _objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("Weight optimisation did not converge: %s", result.message)

        opt_weights = {
            name: float(result.x[i])
            for i, name in enumerate(self.model_names)
        }
        self.weights = self._normalise(opt_weights)
        logger.info("Optimised weights: %s (metric=%s)", self.weights, metric)
        return self.weights

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, user_id: int, movie_id: int) -> float:
        """Weighted-average prediction for a single (user, item) pair.

        Parameters
        ----------
        user_id : int
        movie_id : int

        Returns
        -------
        float
        """
        pair = pd.DataFrame(
            [{"user_id": user_id, "movie_id": movie_id}]
        )
        return float(self.predict_batch(pair)[0])

    def predict_batch(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Weighted-average predictions for many pairs.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns.

        Returns
        -------
        np.ndarray
        """
        preds = self._get_all_predictions(pairs_df)
        blended = np.zeros(len(pairs_df), dtype=np.float64)
        for name in self.model_names:
            blended += self.weights[name] * preds[name]
        return np.clip(blended, 1.0, 5.0).astype(np.float32)

    # ------------------------------------------------------------------
    def recommend_top_k(
        self,
        user_id: int,
        candidate_items: Union[List[int], np.ndarray],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Top-*k* recommendations via weighted ensemble.

        Parameters
        ----------
        user_id : int
        candidate_items : list or array of int
        k : int

        Returns
        -------
        list of (item_id, score)
        """
        items = np.asarray(candidate_items)
        pairs = pd.DataFrame(
            {"user_id": user_id, "movie_id": items}
        )
        scores = self.predict_batch(pairs)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(items[i]), float(scores[i])) for i in top_idx]


# ======================================================================
# Stacking Ensemble
# ======================================================================
class StackingEnsemble:
    """Stacking ensemble with a LightGBM meta-learner.

    The meta-learner is trained on:
    * Predictions from each base model
    * Hand-crafted features: ``user_avg``, ``item_avg``,
      ``n_user_ratings``, ``n_item_ratings``

    Parameters
    ----------
    base_models : dict[str, model]
        Mapping of model names to model objects.  Each must expose
        ``predict_batch(pairs_df)``.
    meta_model : object or None
        Scikit-learn-compatible regressor used as the meta-learner.
        Defaults to ``lightgbm.LGBMRegressor`` with sensible defaults.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_model: Optional[Any] = None,
    ) -> None:
        if not base_models:
            raise ValueError("Must provide at least one base model.")
        self.base_models = base_models
        self.model_names: List[str] = sorted(base_models.keys())

        if meta_model is None:
            try:
                from lightgbm import LGBMRegressor

                self.meta_model = LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                logger.warning(
                    "lightgbm not installed — falling back to "
                    "sklearn.ensemble.GradientBoostingRegressor."
                )
                from sklearn.ensemble import GradientBoostingRegressor

                self.meta_model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                )
        else:
            self.meta_model = meta_model

        # Statistics computed during fit
        self._user_stats: Optional[pd.DataFrame] = None
        self._item_stats: Optional[pd.DataFrame] = None
        self._global_avg: float = 3.0

        logger.info(
            "StackingEnsemble initialised | base_models: %s | meta: %s",
            self.model_names,
            type(self.meta_model).__name__,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _compute_stats(self, df: pd.DataFrame) -> None:
        """Pre-compute user and item statistics from training /
        validation data."""
        self._global_avg = float(df["rating"].mean())
        self._user_stats = (
            df.groupby("user_id")["rating"]
            .agg(user_avg="mean", n_user_ratings="count")
            .reset_index()
        )
        self._item_stats = (
            df.groupby("movie_id")["rating"]
            .agg(item_avg="mean", n_item_ratings="count")
            .reset_index()
        )

    def _build_meta_features(
        self, pairs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build feature matrix for the meta-learner.

        Columns: one per base model prediction + user_avg, item_avg,
        n_user_ratings, n_item_ratings.
        """
        feats = pairs_df[["user_id", "movie_id"]].copy()

        # Base model predictions
        for name in self.model_names:
            feats[f"pred_{name}"] = np.asarray(
                self.base_models[name].predict_batch(pairs_df),
                dtype=np.float32,
            )

        # User stats
        if self._user_stats is not None:
            feats = feats.merge(self._user_stats, on="user_id", how="left")
        else:
            feats["user_avg"] = self._global_avg
            feats["n_user_ratings"] = 0

        # Item stats
        if self._item_stats is not None:
            feats = feats.merge(self._item_stats, on="movie_id", how="left")
        else:
            feats["item_avg"] = self._global_avg
            feats["n_item_ratings"] = 0

        # Fill NaN for cold-start users/items
        feats["user_avg"] = feats["user_avg"].fillna(self._global_avg)
        feats["item_avg"] = feats["item_avg"].fillna(self._global_avg)
        feats["n_user_ratings"] = feats["n_user_ratings"].fillna(0)
        feats["n_item_ratings"] = feats["n_item_ratings"].fillna(0)

        # Drop id columns (not features)
        feature_cols = [
            c for c in feats.columns if c not in ("user_id", "movie_id")
        ]
        return feats[feature_cols]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, val_df: pd.DataFrame) -> "StackingEnsemble":
        """Train the meta-learner on validation data.

        Parameters
        ----------
        val_df : pd.DataFrame
            Columns: ``user_id``, ``movie_id``, ``rating``.  The base
            models must **not** have been trained on this data.

        Returns
        -------
        StackingEnsemble
            ``self`` (for chaining).
        """
        logger.info(
            "Training meta-learner on %d samples …", len(val_df)
        )
        self._compute_stats(val_df)
        X = self._build_meta_features(val_df)
        y = val_df["rating"].values.astype(np.float32)

        self.meta_model.fit(X, y)
        train_preds = self.meta_model.predict(X)
        rmse = float(np.sqrt(np.mean((train_preds - y) ** 2)))
        logger.info("Meta-learner training RMSE: %.4f", rmse)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict a single rating.

        Parameters
        ----------
        user_id : int
        movie_id : int

        Returns
        -------
        float
        """
        pair = pd.DataFrame(
            [{"user_id": user_id, "movie_id": movie_id}]
        )
        return float(self.predict_batch(pair)[0])

    def predict_batch(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for many pairs.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns.

        Returns
        -------
        np.ndarray
        """
        X = self._build_meta_features(pairs_df)
        preds = self.meta_model.predict(X)
        return np.clip(preds, 1.0, 5.0).astype(np.float32)

    # ------------------------------------------------------------------
    def recommend_top_k(
        self,
        user_id: int,
        candidate_items: Union[List[int], np.ndarray],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Top-*k* recommendations via the stacking ensemble.

        Parameters
        ----------
        user_id : int
        candidate_items : list or array of int
        k : int

        Returns
        -------
        list of (item_id, score)
        """
        items = np.asarray(candidate_items)
        pairs = pd.DataFrame(
            {"user_id": user_id, "movie_id": items}
        )
        scores = self.predict_batch(pairs)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(items[i]), float(scores[i])) for i in top_idx]
