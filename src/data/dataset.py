"""
PyTorch Datasets & Utilities for Netflix Prize
===============================================
Provides Dataset classes for pointwise and pairwise (BPR) training, a sparse
interaction-matrix wrapper with graph-model helpers, and a convenience
factory for DataLoaders.

Classes
-------
RatingDataset
    Standard ``(user, item, rating)`` dataset for pointwise losses (MSE, MAE).
PairwiseDataset
    BPR-style ``(user, positive_item, negative_item)`` dataset with online
    negative sampling.
InteractionMatrix
    Sparse CSR matrix wrapping user-item interactions with utility methods
    for neighbourhood lookup and conversion to edge-index tensors.

Functions
---------
create_data_loaders
    One-liner to build train and validation ``DataLoader`` objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import DataLoader, Dataset

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

import pandas as pd

logger = logging.getLogger(__name__)

DataFrame = Union["pl.DataFrame", "pd.DataFrame"]


# ======================================================================
# 1.  Pointwise Dataset
# ======================================================================
class RatingDataset(Dataset):
    """PyTorch Dataset yielding ``(user_id, item_id, rating)`` tensors.

    Parameters
    ----------
    user_ids : np.ndarray
        1-D array of integer user indices.
    item_ids : np.ndarray
        1-D array of integer item (movie) indices.
    ratings : np.ndarray
        1-D array of rating values (int or float).
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
    ) -> None:
        assert len(user_ids) == len(item_ids) == len(ratings), (
            "All input arrays must have the same length."
        )
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings.astype(np.float32))

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

    def __repr__(self) -> str:
        return f"RatingDataset(n_ratings={len(self):,})"


# ======================================================================
# 2.  Pairwise (BPR) Dataset
# ======================================================================
class PairwiseDataset(Dataset):
    """BPR-style dataset with **online** uniform negative sampling.

    For each positive ``(user, pos_item)`` pair the dataset samples a random
    negative item that the user has *not* interacted with.

    Parameters
    ----------
    df : DataFrame
        Interaction DataFrame with ``user_idx`` and ``movie_idx`` columns.
    n_items : int
        Total number of items (movies) in the catalogue — used for
        sampling the negative item uniformly from ``[0, n_items)``.
    """

    def __init__(self, df: DataFrame, n_items: int) -> None:
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)

        if is_polars:
            self.users = df["user_idx"].to_numpy().astype(np.int64)
            self.pos_items = df["movie_idx"].to_numpy().astype(np.int64)
        else:
            self.users = df["user_idx"].values.astype(np.int64)
            self.pos_items = df["movie_idx"].values.astype(np.int64)

        self.n_items = n_items

        # Build per-user positive-item sets for rejection sampling
        self._user_positives: Dict[int, Set[int]] = {}
        for u, i in zip(self.users, self.pos_items):
            self._user_positives.setdefault(int(u), set()).add(int(i))

        logger.info(
            "PairwiseDataset ready — %s interactions, %s users, %s items.",
            f"{len(self.users):,}",
            f"{len(self._user_positives):,}",
            f"{n_items:,}",
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = int(self.users[idx])
        pos_item = int(self.pos_items[idx])

        # Rejection-sample a negative item
        neg_item = np.random.randint(0, self.n_items)
        positives = self._user_positives.get(user, set())
        while neg_item in positives:
            neg_item = np.random.randint(0, self.n_items)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )

    def __repr__(self) -> str:
        return (
            f"PairwiseDataset(n_interactions={len(self):,}, "
            f"n_items={self.n_items:,})"
        )


# ======================================================================
# 3.  Interaction Matrix
# ======================================================================
class InteractionMatrix:
    """Sparse user-item interaction matrix with helper methods.

    Parameters
    ----------
    df : DataFrame
        Must contain ``user_idx``, ``movie_idx``, and ``rating`` columns.
    n_users : int
        Total number of users (determines matrix row count).
    n_items : int
        Total number of items (determines matrix column count).
    binarize : bool
        If *True*, store 1/0 instead of actual ratings (useful for
        implicit-feedback models).
    """

    def __init__(
        self,
        df: DataFrame,
        n_users: int,
        n_items: int,
        binarize: bool = False,
    ) -> None:
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)

        if is_polars:
            rows = df["user_idx"].to_numpy()
            cols = df["movie_idx"].to_numpy()
            vals = df["rating"].to_numpy().astype(np.float32)
        else:
            rows = df["user_idx"].values
            cols = df["movie_idx"].values
            vals = df["rating"].values.astype(np.float32)

        if binarize:
            vals = np.ones_like(vals)

        self.matrix: sp.csr_matrix = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_users, n_items),
        )
        self.n_users = n_users
        self.n_items = n_items

        logger.info(
            "InteractionMatrix built — shape %s, nnz %s (density %.4f%%).",
            self.matrix.shape,
            f"{self.matrix.nnz:,}",
            100.0 * self.matrix.nnz / (n_users * n_items),
        )

    # ----- Neighbourhood queries ----- #
    def get_user_items(self, user_idx: int) -> np.ndarray:
        """Return sorted array of item indices rated by *user_idx*."""
        return self.matrix[user_idx].indices.copy()

    def get_item_users(self, item_idx: int) -> np.ndarray:
        """Return sorted array of user indices who rated *item_idx*."""
        csc = self.matrix.tocsc()
        return csc[:, item_idx].indices.copy()

    def get_user_ratings(self, user_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(item_indices, ratings)`` for a single user."""
        row = self.matrix[user_idx]
        return row.indices.copy(), row.data.copy()

    # ----- Graph helpers ----- #
    def to_edge_index(self) -> torch.Tensor:
        """Convert interactions to a ``[2, E]`` edge-index tensor.

        Suitable for PyTorch Geometric or similar GNN frameworks.
        Returns a **bipartite** edge list where source nodes are users
        (indices ``0 … n_users−1``) and destination nodes are items
        (indices ``n_users … n_users+n_items−1``).

        Returns
        -------
        torch.Tensor
            ``LongTensor`` of shape ``[2, 2*E]`` — includes both
            ``(user → item)`` and ``(item → user)`` edges.
        """
        coo = self.matrix.tocoo()
        user_nodes = torch.from_numpy(coo.row.astype(np.int64))
        # Offset item IDs so they don't collide with user IDs
        item_nodes = torch.from_numpy(coo.col.astype(np.int64)) + self.n_users

        # Bidirectional edges
        src = torch.cat([user_nodes, item_nodes])
        dst = torch.cat([item_nodes, user_nodes])
        edge_index = torch.stack([src, dst], dim=0)

        logger.info(
            "Edge index created — %s edges (bidirectional).",
            f"{edge_index.shape[1]:,}",
        )
        return edge_index

    def to_adjacency_matrix(self, normalize: bool = True) -> sp.csr_matrix:
        """Build the full bipartite adjacency matrix.

        .. math::

            A = \\begin{pmatrix} 0 & R \\\\ R^T & 0 \\end{pmatrix}

        Parameters
        ----------
        normalize : bool
            If *True*, apply symmetric normalisation
            ``D^{-1/2} A D^{-1/2}`` (used by LightGCN, etc.).

        Returns
        -------
        sp.csr_matrix
            Adjacency matrix of shape ``(n_users + n_items, n_users + n_items)``.
        """
        R = self.matrix.astype(np.float32)
        R_binary = (R > 0).astype(np.float32)

        # Build bipartite adjacency
        zeros_users = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zeros_items = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        adj = sp.bmat(
            [[zeros_users, R_binary], [R_binary.T, zeros_items]],
            format="csr",
        )

        if normalize:
            degrees = np.array(adj.sum(axis=1)).flatten()
            d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            adj = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj

    def __repr__(self) -> str:
        return (
            f"InteractionMatrix(n_users={self.n_users:,}, "
            f"n_items={self.n_items:,}, nnz={self.matrix.nnz:,})"
        )


# ======================================================================
# 4.  DataLoader factory
# ======================================================================
def create_data_loaders(
    train_df: DataFrame,
    val_df: DataFrame,
    batch_size: int = 1024,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders for train and validation sets.

    Parameters
    ----------
    train_df : DataFrame
        Training split with ``user_idx``, ``movie_idx``, ``rating`` columns.
    val_df : DataFrame
        Validation split with the same columns.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of data-loading worker processes.
    pin_memory : bool
        Whether to pin tensors in CUDA-pinned memory.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``.
    """
    is_polars = _HAS_POLARS and isinstance(train_df, pl.DataFrame)

    def _extract_arrays(df: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if _HAS_POLARS and isinstance(df, pl.DataFrame):
            return (
                df["user_idx"].to_numpy(),
                df["movie_idx"].to_numpy(),
                df["rating"].to_numpy().astype(np.float32),
            )
        else:
            return (
                df["user_idx"].values,
                df["movie_idx"].values,
                df["rating"].values.astype(np.float32),
            )

    train_u, train_i, train_r = _extract_arrays(train_df)
    val_u, val_i, val_r = _extract_arrays(val_df)

    train_dataset = RatingDataset(train_u, train_i, train_r)
    val_dataset = RatingDataset(val_u, val_i, val_r)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "DataLoaders created — train: %s batches, val: %s batches (batch_size=%d).",
        f"{len(train_loader):,}",
        f"{len(val_loader):,}",
        batch_size,
    )
    return train_loader, val_loader
