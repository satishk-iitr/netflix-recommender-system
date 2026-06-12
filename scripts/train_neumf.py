#!/usr/bin/env python
"""
train_neumf.py — Train a Neural Matrix Factorisation (NeuMF) model.

Usage:
    python scripts/train_neumf.py --data-dir data/processed --output-dir models/neumf
    python scripts/train_neumf.py --data-dir data/processed --output-dir models/neumf \\
        --epochs 30 --batch-size 512 --lr 0.001 --device cuda

Requires:
    pip install torch pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
# Dataset & DataLoader
# ======================================================================

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train / val / test CSVs."""
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, val, test


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    num_workers: int = 0,
):
    """Create PyTorch DataLoaders from DataFrames."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def df_to_dataset(df: pd.DataFrame) -> TensorDataset:
        users = torch.LongTensor(df["user_id"].values)
        items = torch.LongTensor(df["movie_id"].values)
        ratings = torch.FloatTensor(df["rating"].values)
        return TensorDataset(users, items, ratings)

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ======================================================================
# NeuMF Model
# ======================================================================

def build_neumf_model(
    n_users: int,
    n_items: int,
    config: Dict[str, Any],
):
    """Build a NeuMF model (GMF + MLP fusion)."""
    import torch
    import torch.nn as nn

    class NeuMF(nn.Module):
        """Neural Matrix Factorization (He et al., 2017)."""

        def __init__(
            self,
            n_users: int,
            n_items: int,
            gmf_dim: int = 64,
            mlp_dims: Tuple[int, ...] = (128, 64, 32),
            dropout: float = 0.2,
        ):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items

            # GMF pathway
            self.gmf_user_emb = nn.Embedding(n_users, gmf_dim)
            self.gmf_item_emb = nn.Embedding(n_items, gmf_dim)

            # MLP pathway
            mlp_input_dim = gmf_dim * 2  # reuse gmf_dim for MLP embedding
            self.mlp_user_emb = nn.Embedding(n_users, gmf_dim)
            self.mlp_item_emb = nn.Embedding(n_items, gmf_dim)

            mlp_layers = []
            in_dim = mlp_input_dim
            for out_dim in mlp_dims:
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.mlp = nn.Sequential(*mlp_layers)

            # Fusion
            self.output_layer = nn.Linear(gmf_dim + mlp_dims[-1], 1)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, user_ids, item_ids):
            # GMF
            gmf_user = self.gmf_user_emb(user_ids)
            gmf_item = self.gmf_item_emb(item_ids)
            gmf_out = gmf_user * gmf_item  # element-wise product

            # MLP
            mlp_user = self.mlp_user_emb(user_ids)
            mlp_item = self.mlp_item_emb(item_ids)
            mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
            mlp_out = self.mlp(mlp_input)

            # Fusion
            concat = torch.cat([gmf_out, mlp_out], dim=-1)
            rating = self.output_layer(concat).squeeze(-1)
            return rating

        def predict(self, user_id: int, item_id: int) -> float:
            """Single-pair prediction (for evaluation API compatibility)."""
            self.eval()
            device = next(self.parameters()).device
            with torch.no_grad():
                u = torch.LongTensor([user_id]).to(device)
                i = torch.LongTensor([item_id]).to(device)
                return self.forward(u, i).item()

    gmf_dim = config.get("gmf_dim", 64)
    mlp_dims = tuple(config.get("mlp_dims", [128, 64, 32]))
    dropout = config.get("dropout", 0.2)

    model = NeuMF(n_users, n_items, gmf_dim, mlp_dims, dropout)
    logger.info(
        "NeuMF — %d users, %d items, GMF=%d, MLP=%s, dropout=%.2f",
        n_users, n_items, gmf_dim, mlp_dims, dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")
    return model


# ======================================================================
# Training loop
# ======================================================================

def train_neumf(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    patience: int = 5,
) -> Dict[str, Any]:
    """Train NeuMF with early stopping."""
    import torch
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_rmse": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0

        for users, items, ratings in train_loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # --- Validate ---
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(device)
                items = items.to(device)
                preds = model(users, items)
                all_preds.append(preds.cpu().numpy())
                all_true.append(ratings.numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        val_rmse = float(np.sqrt(np.mean((all_true - all_preds) ** 2)))
        val_mae = float(np.mean(np.abs(all_true - all_preds)))

        history["train_loss"].append(avg_train_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        scheduler.step(val_rmse)

        logger.info(
            "Epoch %3d/%d — train_loss=%.4f  val_rmse=%.4f  val_mae=%.4f",
            epoch, epochs, avg_train_loss, val_rmse, val_mae,
        )

        # --- Early stopping ---
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            # Save best checkpoint
            ckpt_path = output_dir / "neumf_best.pt"
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Reload best weights
    best_ckpt = output_dir / "neumf_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        logger.info("Loaded best checkpoint (val_rmse=%.4f)", best_val_rmse)

    return {
        "best_val_rmse": best_val_rmse,
        "final_val_rmse": history["val_rmse"][-1],
        "final_val_mae": history["val_mae"][-1],
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }


# ======================================================================
# Config loading
# ======================================================================

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config if provided."""
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
        description="Train NeuMF on processed Netflix data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
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
    n_users = max(train_df["user_id"].max(), val_df["user_id"].max()) + 1
    n_items = max(train_df["movie_id"].max(), val_df["movie_id"].max()) + 1
    logger.info(
        "Data: %d train, %d val — %d users, %d items",
        len(train_df), len(val_df), n_users, n_items,
    )

    # 2. DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, args.batch_size
    )

    # 3. Build model
    config = load_config(args.config)
    model = build_neumf_model(n_users, n_items, config)

    # 4. Train
    t0 = time.time()
    results = train_neumf(
        model,
        train_loader,
        val_loader,
        epochs=config.get("epochs", args.epochs),
        lr=config.get("lr", args.lr),
        weight_decay=config.get("weight_decay", args.weight_decay),
        device=device,
        output_dir=output_dir,
        patience=config.get("patience", args.patience),
    )
    train_time = time.time() - t0

    # 5. Save
    import torch as th

    final_path = output_dir / "neumf_model.pt"
    th.save(model.state_dict(), final_path)

    # Also save model config for reloading
    model_meta = {
        "n_users": n_users,
        "n_items": n_items,
        "config": config,
        "train_time_s": round(train_time, 2),
        **{k: v for k, v in results.items() if k != "history"},
    }
    with open(output_dir / "neumf_results.json", "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, default=str)

    logger.info("NeuMF training complete ✓  Best val RMSE=%.4f", results["best_val_rmse"])


if __name__ == "__main__":
    main()
