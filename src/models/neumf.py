"""
Neural Matrix Factorization (NeuMF) Model
==========================================

Implements the NeuMF architecture from He et al. (2017) combining Generalized
Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) for collaborative
filtering. Adapted for the Netflix Prize rating prediction task (regression).

Architecture
------------
1. **GMF branch**: user/item embeddings → element-wise product
2. **MLP branch**: user/item embeddings → concat → FC(128→64→32) w/ ReLU,
   BatchNorm, Dropout
3. **Fusion**: concat GMF & MLP outputs → linear → sigmoid → scale to [1, 5]

Example
-------
>>> model = NeuMF(n_users=480189, n_items=17770, embed_dim=64)
>>> trainer = NeuMFTrainer(model, device='cuda')
>>> history = trainer.fit(train_loader, val_loader, epochs=20)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ======================================================================
# Model
# ======================================================================
class NeuMF(nn.Module):
    """Neural Matrix Factorization combining GMF and MLP branches.

    Parameters
    ----------
    n_users : int
        Total number of users (determines embedding table size).
    n_items : int
        Total number of items.
    embed_dim : int
        Embedding dimension for the GMF branch (and the base dim for MLP
        branch which uses ``embed_dim`` as well).
    mlp_layers : list of int
        Hidden layer sizes for the MLP tower. Defaults to ``[128, 64, 32]``.
    dropout : float
        Dropout probability applied after each MLP hidden layer.
    rating_range : tuple of float
        ``(min, max)`` range for final prediction clipping.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        rating_range: Tuple[float, float] = (1.0, 5.0),
    ) -> None:
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.rating_range = rating_range
        mlp_layers = mlp_layers or [128, 64, 32]

        # ---- GMF branch embeddings ----
        self.user_embed_gmf = nn.Embedding(n_users, embed_dim)
        self.item_embed_gmf = nn.Embedding(n_items, embed_dim)

        # ---- MLP branch embeddings ----
        self.user_embed_mlp = nn.Embedding(n_users, embed_dim)
        self.item_embed_mlp = nn.Embedding(n_items, embed_dim)

        # ---- MLP tower ----
        mlp_input_dim = embed_dim * 2  # concat of user + item embeddings
        layers: list[nn.Module] = []
        for out_dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # ---- Fusion ----
        # GMF output dim = embed_dim; MLP output dim = mlp_layers[-1]
        fusion_dim = embed_dim + mlp_layers[-1]
        self.fc_out = nn.Linear(fusion_dim, 1)

        # ---- Initialisation (Xavier) ----
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all parameters."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning predicted ratings.

        Parameters
        ----------
        user_ids : LongTensor of shape ``(batch,)``
        item_ids : LongTensor of shape ``(batch,)``

        Returns
        -------
        Tensor of shape ``(batch,)``
            Predicted ratings clipped to ``self.rating_range``.
        """
        # GMF branch
        gmf_user = self.user_embed_gmf(user_ids)  # (B, D)
        gmf_item = self.item_embed_gmf(item_ids)  # (B, D)
        gmf_out = gmf_user * gmf_item              # element-wise product

        # MLP branch
        mlp_user = self.user_embed_mlp(user_ids)
        mlp_item = self.item_embed_mlp(item_ids)
        mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.fc_out(fused).squeeze(-1)  # (B,)

        # Scale sigmoid output to rating range
        lo, hi = self.rating_range
        pred = torch.sigmoid(logit) * (hi - lo) + lo
        return pred


# ======================================================================
# Trainer
# ======================================================================
class NeuMFTrainer:
    """Training harness for :class:`NeuMF`.

    Parameters
    ----------
    model : NeuMF
        The NeuMF model instance.
    device : str or torch.device
        Target device (e.g. ``'cuda'`` or ``'cpu'``).
    lr : float
        Learning rate for Adam optimiser.
    weight_decay : float
        L2 regularisation coefficient.
    """

    def __init__(
        self,
        model: NeuMF,
        device: Union[str, torch.device] = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.best_state: Optional[Dict[str, Any]] = None
        logger.info(
            "NeuMFTrainer initialised | device=%s | lr=%.1e | wd=%.1e",
            self.device,
            lr,
            weight_decay,
        )

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Yields batches of ``(user_ids, item_ids, ratings)`` tensors.

        Returns
        -------
        float
            Average MSE loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for user_ids, item_ids, ratings in tqdm(
            train_loader, desc="  train", leave=False
        ):
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.float().to(self.device)

            preds = self.model(user_ids, item_ids)
            loss = self.criterion(preds, ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Compute RMSE on a validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Yields ``(user_ids, item_ids, ratings)`` tensors.

        Returns
        -------
        float
            RMSE on the validation data.
        """
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for user_ids, item_ids, ratings in val_loader:
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            preds = self.model(user_ids, item_ids)
            all_preds.append(preds.cpu())
            all_labels.append(ratings.float())

        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        rmse = torch.sqrt(torch.mean((preds_cat - labels_cat) ** 2)).item()
        return rmse

    # ------------------------------------------------------------------
    # Full training loop with early stopping
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        patience: int = 5,
    ) -> Dict[str, list]:
        """Train the model with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        epochs : int
            Maximum number of training epochs.
        patience : int
            Number of epochs with no improvement before stopping.

        Returns
        -------
        dict
            ``{'train_loss': [...], 'val_rmse': [...]}``
        """
        history: Dict[str, list] = {"train_loss": [], "val_rmse": []}
        best_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_rmse = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_rmse"].append(val_rmse)

            improved = val_rmse < best_rmse
            marker = " ★" if improved else ""
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_rmse=%.4f%s",
                epoch,
                epochs,
                train_loss,
                val_rmse,
                marker,
            )
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_rmse:.4f}{marker}"
            )

            if improved:
                best_rmse = val_rmse
                patience_counter = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best RMSE=%.4f)",
                        epoch,
                        best_rmse,
                    )
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best RMSE={best_rmse:.4f})"
                    )
                    break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)
            logger.info("Restored best model weights (RMSE=%.4f).", best_rmse)

        return history

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        user_ids: Union[np.ndarray, Sequence[int]],
        item_ids: Union[np.ndarray, Sequence[int]],
    ) -> np.ndarray:
        """Predict ratings for arrays of user and item ids.

        Parameters
        ----------
        user_ids : array-like of int
        item_ids : array-like of int

        Returns
        -------
        np.ndarray
            Predicted ratings of shape ``(N,)``.
        """
        self.model.eval()
        u = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        i = torch.tensor(item_ids, dtype=torch.long, device=self.device)
        preds = self.model(u, i)
        return preds.cpu().numpy()

    def predict_batch(self, pairs_df: pd.DataFrame) -> np.ndarray:
        """Predict ratings from a DataFrame with ``user_id``, ``movie_id``
        columns.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns.

        Returns
        -------
        np.ndarray
        """
        return self.predict(
            pairs_df["user_id"].values, pairs_df["movie_id"].values
        )

    @torch.no_grad()
    def recommend_top_k(
        self,
        user_id: int,
        candidate_items: Union[List[int], np.ndarray],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return top-*k* recommendations for a single user.

        Parameters
        ----------
        user_id : int
        candidate_items : list or array of int
        k : int

        Returns
        -------
        list of (item_id, predicted_score)
        """
        self.model.eval()
        items = np.asarray(candidate_items)
        users = np.full(len(items), user_id, dtype=np.int64)

        scores = self.predict(users, items)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(items[i]), float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save model weights and trainer state.

        Parameters
        ----------
        path : str or Path
            File path (e.g. ``'models/neumf.pt'``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "model_config": {
                "n_users": self.model.n_users,
                "n_items": self.model.n_items,
                "embed_dim": self.model.embed_dim,
                "rating_range": self.model.rating_range,
            },
            "optimizer_state": self.optimizer.state_dict(),
            "best_state": self.best_state,
        }
        torch.save(checkpoint, path)
        logger.info("NeuMF checkpoint saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> "NeuMFTrainer":
        """Load a saved checkpoint and return a ready-to-use trainer.

        Parameters
        ----------
        path : str or Path
        device : str or torch.device
        lr : float
        weight_decay : float

        Returns
        -------
        NeuMFTrainer
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["model_config"]
        model = NeuMF(**config)
        model.load_state_dict(checkpoint["model_state"])

        trainer = cls(model, device=device, lr=lr, weight_decay=weight_decay)
        if checkpoint.get("optimizer_state"):
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        trainer.best_state = checkpoint.get("best_state")
        logger.info("NeuMF checkpoint loaded from %s", path)
        return trainer
