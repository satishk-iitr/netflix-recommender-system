"""
LightGCN – Light Graph Convolution Network
============================================

Implements the LightGCN model from He et al. (2020) for collaborative
filtering on the Netflix Prize dataset. LightGCN simplifies GCN by
removing feature transformation and nonlinear activation, learning user
and item embeddings purely through neighbourhood aggregation on the
user–item bipartite graph.

Key design choices
------------------
* **No feature transformation / no activation** — only neighbourhood
  averaging at each layer.
* **Layer combination** — the final embedding is the mean of embeddings
  from all layers (including layer 0).
* **BPR loss** — trained with Bayesian Personalised Ranking loss plus
  L2 embedding regularisation.

Example
-------
>>> from src.models.lightgcn import LightGCN, LightGCNTrainer
>>> edge_index = LightGCNTrainer.build_edge_index(train_df, n_users)
>>> model = LightGCN(n_users, n_items, embed_dim=64, n_layers=3)
>>> trainer = LightGCNTrainer(model, edge_index, device='cuda')
>>> trainer.fit(train_df, val_df, epochs=50, batch_size=2048)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ======================================================================
# Model
# ======================================================================
class LightGCN(nn.Module):
    """Light Graph Convolution Network.

    Parameters
    ----------
    n_users : int
        Number of users in the graph.
    n_items : int
        Number of items in the graph.
    embed_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of graph convolution layers.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Learnable embeddings (users first, then items)
        self.embedding = nn.Embedding(self.n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    # ------------------------------------------------------------------
    def forward(
        self, adj: torch.sparse.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run light graph convolution and return final user / item
        embeddings.

        Parameters
        ----------
        adj : torch.sparse.FloatTensor
            Symmetrically-normalised adjacency matrix of shape
            ``(n_nodes, n_nodes)``.

        Returns
        -------
        user_embeds : Tensor of shape ``(n_users, embed_dim)``
        item_embeds : Tensor of shape ``(n_items, embed_dim)``
        """
        all_embeds = self.embedding.weight  # (N, D)
        layer_embeds = [all_embeds]

        for _ in range(self.n_layers):
            all_embeds = torch.sparse.mm(adj, all_embeds)
            layer_embeds.append(all_embeds)

        # Layer combination: simple mean
        stacked = torch.stack(layer_embeds, dim=0)  # (L+1, N, D)
        final = stacked.mean(dim=0)  # (N, D)

        user_embeds = final[: self.n_users]
        item_embeds = final[self.n_users :]
        return user_embeds, item_embeds

    # ------------------------------------------------------------------
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_embeds: torch.Tensor,
        item_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted scores (inner product).

        Parameters
        ----------
        user_ids : LongTensor ``(B,)``
        item_ids : LongTensor ``(B,)``
        user_embeds : Tensor ``(n_users, D)``
        item_embeds : Tensor ``(n_items, D)``

        Returns
        -------
        Tensor ``(B,)``
        """
        u = user_embeds[user_ids]
        i = item_embeds[item_ids]
        return (u * i).sum(dim=-1)

    # ------------------------------------------------------------------
    def bpr_loss(
        self,
        user_embeds: torch.Tensor,
        item_embeds: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> torch.Tensor:
        """Bayesian Personalised Ranking loss with L2 regularisation.

        Parameters
        ----------
        user_embeds : Tensor ``(n_users, D)``
        item_embeds : Tensor ``(n_items, D)``
        users : LongTensor ``(B,)``
        pos_items : LongTensor ``(B,)``
        neg_items : LongTensor ``(B,)``
        reg_weight : float
            L2 regularisation coefficient on raw embeddings.

        Returns
        -------
        Tensor (scalar)
        """
        u_emb = user_embeds[users]
        pos_emb = item_embeds[pos_items]
        neg_emb = item_embeds[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)

        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 reg on the *initial* (layer-0) embeddings, not the propagated ones
        u_raw = self.embedding.weight[users]
        pi_raw = self.embedding.weight[self.n_users + pos_items]
        ni_raw = self.embedding.weight[self.n_users + neg_items]
        l2 = (
            u_raw.norm(2).pow(2)
            + pi_raw.norm(2).pow(2)
            + ni_raw.norm(2).pow(2)
        ) / (2 * len(users))

        return bpr + reg_weight * l2


# ======================================================================
# Trainer
# ======================================================================
class LightGCNTrainer:
    """Training harness for :class:`LightGCN`.

    Parameters
    ----------
    model : LightGCN
    edge_index : torch.LongTensor of shape ``(2, E)``
        Bidirectional edges in the user–item bipartite graph.  User
        node ids are in ``[0, n_users)`` and item node ids are in
        ``[n_users, n_users + n_items)``.
    device : str or torch.device
    lr : float
    weight_decay : float
    """

    def __init__(
        self,
        model: LightGCN,
        edge_index: torch.LongTensor,
        device: Union[str, torch.device] = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Build sparse normalised adjacency once
        self.adj = self._build_norm_adj(edge_index, model.n_nodes).to(
            self.device
        )
        self.best_state: Optional[Dict[str, Any]] = None
        # Keep a copy of the edge index for serialisation
        self._edge_index = edge_index

        logger.info(
            "LightGCNTrainer ready | nodes=%d | edges=%d | device=%s",
            model.n_nodes,
            edge_index.shape[1],
            self.device,
        )

    # ------------------------------------------------------------------
    # Adjacency helpers
    # ------------------------------------------------------------------
    @staticmethod
    def build_edge_index(
        train_df: pd.DataFrame, n_users: int
    ) -> torch.LongTensor:
        """Build a bidirectional edge index from a training DataFrame.

        User ids are kept as-is (``[0, n_users)``).  Item ids are offset
        by ``n_users`` so that user and item node sets are disjoint.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns with
            **0-indexed** ids.
        n_users : int
            Total number of users (used as offset for item node ids).

        Returns
        -------
        torch.LongTensor of shape ``(2, 2*E)``
            Bidirectional edges.
        """
        users = train_df["user_id"].values.astype(np.int64)
        items = train_df["movie_id"].values.astype(np.int64) + n_users

        # Bidirectional
        src = np.concatenate([users, items])
        dst = np.concatenate([items, users])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        return edge_index

    @staticmethod
    def _build_norm_adj(
        edge_index: torch.LongTensor, n_nodes: int
    ) -> torch.sparse.FloatTensor:
        """Build a symmetrically normalised sparse adjacency matrix
        ``D^{-1/2} A D^{-1/2}``.

        Parameters
        ----------
        edge_index : LongTensor ``(2, E)``
        n_nodes : int

        Returns
        -------
        torch.sparse.FloatTensor
        """
        row, col = edge_index[0], edge_index[1]
        values = torch.ones(edge_index.shape[1], dtype=torch.float32)

        # Compute degree
        deg = torch.zeros(n_nodes, dtype=torch.float32)
        deg.scatter_add_(0, row, values.clone())

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Normalised values
        norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(
            edge_index, norm_values, size=(n_nodes, n_nodes)
        ).coalesce()
        return adj

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_negatives(
        users: np.ndarray,
        n_items: int,
        pos_items: np.ndarray,
    ) -> np.ndarray:
        """Uniform negative sampling (one neg per positive).

        Parameters
        ----------
        users : array of int (unused here but kept for API symmetry)
        n_items : int
        pos_items : array of int

        Returns
        -------
        np.ndarray of negative item ids
        """
        neg_items = np.random.randint(0, n_items, size=len(pos_items))
        # Re-sample collisions (pos == neg).  Rare for large catalogues.
        collision = neg_items == pos_items
        while collision.any():
            neg_items[collision] = np.random.randint(
                0, n_items, size=collision.sum()
            )
            collision = neg_items == pos_items
        return neg_items

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        train_interactions: pd.DataFrame,
        batch_size: int = 2048,
        n_items: Optional[int] = None,
    ) -> float:
        """Run one BPR training epoch.

        Parameters
        ----------
        train_interactions : pd.DataFrame
            Must contain ``user_id`` and ``movie_id`` columns (0-indexed).
        batch_size : int
        n_items : int or None
            Total item count (for negative sampling).  Inferred from
            model if *None*.

        Returns
        -------
        float
            Average BPR loss over the epoch.
        """
        self.model.train()
        n_items = n_items or self.model.n_items

        users = train_interactions["user_id"].values
        pos_items = train_interactions["movie_id"].values
        n = len(users)

        # Shuffle
        perm = np.random.permutation(n)
        users = users[perm]
        pos_items = pos_items[perm]

        total_loss = 0.0
        n_batches = 0

        for start in tqdm(
            range(0, n, batch_size), desc="  train", leave=False
        ):
            end = min(start + batch_size, n)
            batch_users = users[start:end]
            batch_pos = pos_items[start:end]
            batch_neg = self._sample_negatives(batch_users, n_items, batch_pos)

            u_t = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            p_t = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
            n_t = torch.tensor(batch_neg, dtype=torch.long, device=self.device)

            user_embeds, item_embeds = self.model(self.adj)
            loss = self.model.bpr_loss(
                user_embeds, item_embeds, u_t, p_t, n_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        val_df: pd.DataFrame,
        user_embeds: Optional[torch.Tensor] = None,
        item_embeds: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute RMSE on validation data.

        The inner-product scores are **not** directly on the 1–5 scale
        (LightGCN is trained with BPR). We still report RMSE for
        monitoring, but downstream stacking / calibration layers should
        handle scale alignment.

        Parameters
        ----------
        val_df : pd.DataFrame
            Columns: ``user_id``, ``movie_id``, ``rating``.
        user_embeds, item_embeds : Tensor, optional
            Pre-computed embeddings.  If *None*, a forward pass is run.

        Returns
        -------
        float
        """
        self.model.eval()
        if user_embeds is None or item_embeds is None:
            user_embeds, item_embeds = self.model(self.adj)

        u = torch.tensor(
            val_df["user_id"].values, dtype=torch.long, device=self.device
        )
        i = torch.tensor(
            val_df["movie_id"].values, dtype=torch.long, device=self.device
        )
        preds = self.model.predict(u, i, user_embeds, item_embeds)
        # Clip to rating scale for RMSE computation
        preds = preds.clamp(1.0, 5.0)
        targets = torch.tensor(
            val_df["rating"].values, dtype=torch.float32, device=self.device
        )
        rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
        return rmse

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 2048,
        patience: int = 5,
    ) -> Dict[str, list]:
        """Full training loop with early stopping.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions (``user_id``, ``movie_id``, ``rating``).
        val_df : pd.DataFrame
            Validation interactions.
        epochs : int
        batch_size : int
        patience : int

        Returns
        -------
        dict
            ``{'train_loss': [...], 'val_rmse': [...]}``
        """
        history: Dict[str, list] = {"train_loss": [], "val_rmse": []}
        best_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_df, batch_size=batch_size)
            val_rmse = self.evaluate(val_df)

            history["train_loss"].append(loss)
            history["val_rmse"].append(val_rmse)

            improved = val_rmse < best_rmse
            marker = " ★" if improved else ""
            logger.info(
                "Epoch %3d/%d | bpr_loss=%.4f | val_rmse=%.4f%s",
                epoch,
                epochs,
                loss,
                val_rmse,
                marker,
            )
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"bpr_loss={loss:.4f} | "
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

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def recommend_top_k(
        self,
        user_id: int,
        candidate_items: Union[List[int], np.ndarray],
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Return top-*k* items for a given user.

        Parameters
        ----------
        user_id : int
        candidate_items : list or array of int  (0-indexed item ids)
        k : int

        Returns
        -------
        list of (item_id, score)
        """
        self.model.eval()
        user_embeds, item_embeds = self.model(self.adj)

        items = np.asarray(candidate_items)
        u_t = torch.tensor(
            [user_id] * len(items), dtype=torch.long, device=self.device
        )
        i_t = torch.tensor(items, dtype=torch.long, device=self.device)
        scores = self.model.predict(u_t, i_t, user_embeds, item_embeds)
        scores_np = scores.cpu().numpy()

        top_idx = np.argsort(scores_np)[::-1][:k]
        return [(int(items[i]), float(scores_np[i])) for i in top_idx]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save the model checkpoint.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.model.state_dict(),
            "model_config": {
                "n_users": self.model.n_users,
                "n_items": self.model.n_items,
                "embed_dim": self.model.embed_dim,
                "n_layers": self.model.n_layers,
            },
            "optimizer_state": self.optimizer.state_dict(),
            "best_state": self.best_state,
            "edge_index": self._edge_index,
        }
        torch.save(checkpoint, path)
        logger.info("LightGCN checkpoint saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> "LightGCNTrainer":
        """Load a checkpoint and return a trainer.

        Parameters
        ----------
        path : str or Path
        device : str or torch.device
        lr : float
        weight_decay : float

        Returns
        -------
        LightGCNTrainer
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["model_config"]
        model = LightGCN(**config)
        model.load_state_dict(checkpoint["model_state"])

        edge_index = checkpoint["edge_index"]
        trainer = cls(
            model,
            edge_index=edge_index,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
        )
        if checkpoint.get("optimizer_state"):
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        trainer.best_state = checkpoint.get("best_state")
        logger.info("LightGCN checkpoint loaded from %s", path)
        return trainer
