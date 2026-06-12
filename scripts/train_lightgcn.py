#!/usr/bin/env python
"""
train_lightgcn.py — Train a LightGCN graph-based collaborative filtering model.

Usage:
    python scripts/train_lightgcn.py --data-dir data/processed --output-dir models/lightgcn
    python scripts/train_lightgcn.py --data-dir data/processed --output-dir models/lightgcn \\
        --epochs 50 --batch-size 2048 --lr 0.001 --device cuda

Requires:
    pip install torch pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Data helpers
# ======================================================================

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train / val / test CSVs."""
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, val, test


def build_adj_matrix(train_df: pd.DataFrame, n_users: int, n_items: int):
    """Build normalised bipartite adjacency matrix for LightGCN.

    The adjacency matrix is of shape (n_users + n_items, n_users + n_items).
    Normalisation: D^{-1/2} A D^{-1/2}  (symmetric).

    Returns a sparse torch tensor.
    """
    import torch
    from scipy.sparse import coo_matrix, eye

    user_ids = train_df["user_id"].values
    item_ids = train_df["movie_id"].values + n_users  # offset items

    n_nodes = n_users + n_items

    # Build bipartite edges (both directions)
    rows = np.concatenate([user_ids, item_ids])
    cols = np.concatenate([item_ids, user_ids])
    data = np.ones(len(rows), dtype=np.float32)

    adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Symmetric normalisation
    degree = np.array(adj.sum(axis=1)).flatten()
    degree_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
    D_inv_sqrt = coo_matrix(
        (degree_inv_sqrt, (np.arange(n_nodes), np.arange(n_nodes))),
        shape=(n_nodes, n_nodes),
    )
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()

    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size([n_nodes, n_nodes]))

    logger.info(
        "Adjacency matrix: %d nodes, %d edges (normalised)",
        n_nodes, len(norm_adj.data),
    )
    return adj_tensor


# ======================================================================
# LightGCN Model
# ======================================================================

def build_lightgcn_model(
    n_users: int,
    n_items: int,
    config: Dict[str, Any],
):
    """Build a LightGCN model (He et al., 2020)."""
    import torch
    import torch.nn as nn

    class LightGCN(nn.Module):
        """Light Graph Convolution Network for collaborative filtering.

        Simplifies GCN by removing feature transformation and nonlinear
        activation, using only neighbourhood aggregation + layer
        combination.
        """

        def __init__(
            self,
            n_users: int,
            n_items: int,
            emb_dim: int = 64,
            n_layers: int = 3,
        ):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.emb_dim = emb_dim
            self.n_layers = n_layers

            self.user_embedding = nn.Embedding(n_users, emb_dim)
            self.item_embedding = nn.Embedding(n_items, emb_dim)

            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

            # Placeholder — set via set_adj()
            self.adj: Optional[torch.Tensor] = None

        def set_adj(self, adj: torch.Tensor) -> None:
            """Set the normalised adjacency matrix (sparse tensor)."""
            self.adj = adj

        def _propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
            """Run multi-layer graph convolution; return final user & item embeddings."""
            all_emb = torch.cat(
                [self.user_embedding.weight, self.item_embedding.weight], dim=0
            )  # (n_nodes, emb_dim)

            layer_embeddings = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.adj, all_emb)
                layer_embeddings.append(all_emb)

            # Mean of all layers (layer combination)
            final_emb = torch.stack(layer_embeddings, dim=0).mean(dim=0)

            user_emb = final_emb[: self.n_users]
            item_emb = final_emb[self.n_users :]
            return user_emb, item_emb

        def forward(
            self, user_ids: torch.Tensor, item_ids: torch.Tensor
        ) -> torch.Tensor:
            user_emb, item_emb = self._propagate()
            u = user_emb[user_ids]
            i = item_emb[item_ids]
            return (u * i).sum(dim=-1)

        def predict(self, user_id: int, item_id: int) -> float:
            """Single-pair prediction for evaluation API compatibility."""
            self.eval()
            device = self.user_embedding.weight.device
            with torch.no_grad():
                u = torch.LongTensor([user_id]).to(device)
                i = torch.LongTensor([item_id]).to(device)
                return self.forward(u, i).item()

        def bpr_loss(
            self,
            user_ids: torch.Tensor,
            pos_item_ids: torch.Tensor,
            neg_item_ids: torch.Tensor,
            reg_weight: float = 1e-5,
        ) -> torch.Tensor:
            """BPR pairwise loss with L2 regularisation."""
            user_emb, item_emb = self._propagate()

            u = user_emb[user_ids]
            pos = item_emb[pos_item_ids]
            neg = item_emb[neg_item_ids]

            pos_scores = (u * pos).sum(dim=-1)
            neg_scores = (u * neg).sum(dim=-1)

            bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            # L2 regularisation on initial embeddings
            reg = reg_weight * (
                self.user_embedding.weight[user_ids].norm(2).pow(2)
                + self.item_embedding.weight[pos_item_ids].norm(2).pow(2)
                + self.item_embedding.weight[neg_item_ids].norm(2).pow(2)
            ) / len(user_ids)

            return bpr + reg

    emb_dim = config.get("emb_dim", 64)
    n_layers = config.get("n_layers", 3)

    model = LightGCN(n_users, n_items, emb_dim, n_layers)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LightGCN — %d users, %d items, dim=%d, layers=%d, params=%s",
        n_users, n_items, emb_dim, n_layers, f"{total_params:,}",
    )
    return model


# ======================================================================
# Training loop (BPR)
# ======================================================================

def sample_negatives(
    train_df: pd.DataFrame, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample one negative item per interaction (uniform random)."""
    user_item_set = set(
        zip(train_df["user_id"].values, train_df["movie_id"].values)
    )
    neg_items = np.empty(len(train_df), dtype=np.int64)
    for idx, (uid, _iid) in enumerate(
        zip(train_df["user_id"].values, train_df["movie_id"].values)
    ):
        while True:
            neg = rng.integers(0, n_items)
            if (uid, neg) not in user_item_set:
                neg_items[idx] = neg
                break
    return neg_items


def train_lightgcn(
    model,
    adj: "torch.Tensor",
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_items: int,
    epochs: int,
    lr: float,
    reg_weight: float,
    batch_size: int,
    device: str,
    output_dir: Path,
    patience: int = 5,
) -> Dict[str, Any]:
    """Train LightGCN with BPR loss and early stopping on val RMSE."""
    import torch

    model = model.to(device)
    adj = adj.to(device)
    model.set_adj(adj)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    rng = np.random.default_rng(42)
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()

        # Sample negatives for this epoch
        neg_items = sample_negatives(train_df, n_items, rng)

        users = train_df["user_id"].values
        pos_items = train_df["movie_id"].values

        # Shuffle
        perm = rng.permutation(len(users))
        users = users[perm]
        pos_items = pos_items[perm]
        neg_items = neg_items[perm]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            u_batch = torch.LongTensor(users[start:end]).to(device)
            pos_batch = torch.LongTensor(pos_items[start:end]).to(device)
            neg_batch = torch.LongTensor(neg_items[start:end]).to(device)

            optimizer.zero_grad()
            loss = model.bpr_loss(u_batch, pos_batch, neg_batch, reg_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        history["train_loss"].append(avg_loss)

        # --- Validate (RMSE on rating prediction — approximate) ---
        model.eval()
        with torch.no_grad():
            val_users = torch.LongTensor(val_df["user_id"].values).to(device)
            val_items = torch.LongTensor(val_df["movie_id"].values).to(device)
            val_preds = model(val_users, val_items).cpu().numpy()
            val_true = val_df["rating"].values
            val_rmse = float(np.sqrt(np.mean((val_true - val_preds) ** 2)))

        history["val_rmse"].append(val_rmse)
        scheduler.step(val_rmse)

        logger.info(
            "Epoch %3d/%d — bpr_loss=%.4f  val_rmse=%.4f",
            epoch, epochs, avg_loss, val_rmse,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "lightgcn_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Reload best
    best_ckpt = output_dir / "lightgcn_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(
            torch.load(best_ckpt, map_location=device, weights_only=True)
        )
        model.set_adj(adj)

    return {
        "best_val_rmse": best_val_rmse,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }


# ======================================================================
# Config loading
# ======================================================================

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed; ignoring --config")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return cfg or {}


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGCN on processed Netflix data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    # 1. Load data
    train_df, val_df, test_df = load_data(data_dir)
    n_users = max(
        train_df["user_id"].max(), val_df["user_id"].max(), test_df["user_id"].max()
    ) + 1
    n_items = max(
        train_df["movie_id"].max(), val_df["movie_id"].max(), test_df["movie_id"].max()
    ) + 1
    logger.info(
        "Data: %d train, %d val — %d users, %d items",
        len(train_df), len(val_df), n_users, n_items,
    )

    # 2. Build adjacency
    adj = build_adj_matrix(train_df, n_users, n_items)

    # 3. Build model
    config = load_config(args.config)
    model = build_lightgcn_model(n_users, n_items, config)

    # 4. Train
    t0 = time.time()
    results = train_lightgcn(
        model,
        adj,
        train_df,
        val_df,
        n_items,
        epochs=config.get("epochs", args.epochs),
        lr=config.get("lr", args.lr),
        reg_weight=config.get("reg_weight", args.reg_weight),
        batch_size=config.get("batch_size", args.batch_size),
        device=device,
        output_dir=output_dir,
        patience=config.get("patience", args.patience),
    )
    train_time = time.time() - t0

    # 5. Save
    torch.save(model.state_dict(), output_dir / "lightgcn_model.pt")

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "config": config,
        "train_time_s": round(train_time, 2),
        **{k: v for k, v in results.items() if k != "history"},
    }
    with open(output_dir / "lightgcn_results.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(
        "LightGCN training complete ✓  Best val RMSE=%.4f", results["best_val_rmse"]
    )


if __name__ == "__main__":
    main()
