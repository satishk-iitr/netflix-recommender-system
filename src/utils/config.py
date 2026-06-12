"""
Project-wide configuration and hyperparameter management.

Centralises every path, constant, and default hyperparameter so that the
rest of the codebase can simply ``from src.utils.config import Config``.
Model-specific overrides can be loaded from YAML files under ``configs/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Auto-detect project root:  config.py → utils/ → src/ → ROOT
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
CONFIGS_DIR: Path = ROOT_DIR / "configs"


def ensure_dirs() -> None:
    """Create every project directory that doesn't exist yet."""
    for d in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        CONFIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

RATING_SCALE: Tuple[int, int] = (1, 5)
RELEVANCE_THRESHOLD: float = 3.5
TOP_K: int = 10
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.2


# ---------------------------------------------------------------------------
# Default hyper-parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SVDConfig:
    """Hyper-parameters for Surprise SVD."""

    n_factors: int = 150
    n_epochs: int = 30
    lr_all: float = 0.005
    reg_all: float = 0.02
    biased: bool = True
    random_state: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class NeuMFConfig:
    """Hyper-parameters for Neural Matrix Factorization."""

    embed_dim: int = 64
    mlp_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 20
    batch_size: int = 1024
    neg_samples: int = 4
    random_state: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class LightGCNConfig:
    """Hyper-parameters for LightGCN."""

    embed_dim: int = 64
    n_layers: int = 3
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 2048
    reg_weight: float = 1e-4
    random_state: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Unified Config facade
# ---------------------------------------------------------------------------


class Config:
    """
    Central configuration hub.

    Usage
    -----
    >>> cfg = Config()
    >>> cfg.svd.n_factors
    150
    >>> cfg = Config.from_yaml("configs/svd.yaml")  # override from file
    """

    # Paths ------------------------------------------------------------------
    ROOT_DIR: Path = ROOT_DIR
    DATA_DIR: Path = DATA_DIR
    DATA_RAW_DIR: Path = DATA_RAW_DIR
    DATA_PROCESSED_DIR: Path = DATA_PROCESSED_DIR
    RESULTS_DIR: Path = RESULTS_DIR
    FIGURES_DIR: Path = FIGURES_DIR
    PREDICTIONS_DIR: Path = PREDICTIONS_DIR
    CONFIGS_DIR: Path = CONFIGS_DIR

    # Constants --------------------------------------------------------------
    RATING_SCALE: Tuple[int, int] = RATING_SCALE
    RELEVANCE_THRESHOLD: float = RELEVANCE_THRESHOLD
    TOP_K: int = TOP_K
    RANDOM_SEED: int = RANDOM_SEED
    TRAIN_RATIO: float = TRAIN_RATIO
    VAL_RATIO: float = VAL_RATIO
    TEST_RATIO: float = TEST_RATIO

    def __init__(self) -> None:
        self.svd = SVDConfig()
        self.neumf = NeuMFConfig()
        self.lightgcn = LightGCNConfig()

    # --- YAML loaders -------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Read a YAML file and return its contents as a dict."""
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """
        Build a Config from a YAML file, overriding only the keys present.

        The YAML file should have a top-level ``model`` key whose value is
        one of ``svd``, ``neumf``, or ``lightgcn``, followed by the
        hyper-parameter keys to override.
        """
        cfg = cls()
        data = cls._load_yaml(Path(yaml_path))
        model_name = data.get("model", "").lower()
        params = data.get("params", data)

        if model_name == "svd":
            cfg.svd = SVDConfig(**{k: v for k, v in params.items() if k != "model"})
        elif model_name == "neumf":
            cfg.neumf = NeuMFConfig(**{k: v for k, v in params.items() if k != "model"})
        elif model_name == "lightgcn":
            cfg.lightgcn = LightGCNConfig(
                **{k: v for k, v in params.items() if k != "model"}
            )

        return cfg

    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load config for a specific model from the configs/ directory.

        Parameters
        ----------
        model_name : str
            One of ``"svd"``, ``"neumf"``, or ``"lightgcn"``.

        Returns
        -------
        dict
            Merged hyper-parameters (defaults + YAML overrides).
        """
        yaml_path = self.CONFIGS_DIR / f"{model_name}.yaml"
        if yaml_path.exists():
            overrides = self._load_yaml(yaml_path).get("params", {})
        else:
            overrides = {}

        defaults = getattr(self, model_name).to_dict()
        defaults.update(overrides)
        return defaults

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Config(\n"
            f"  ROOT_DIR={self.ROOT_DIR},\n"
            f"  svd={self.svd},\n"
            f"  neumf={self.neumf},\n"
            f"  lightgcn={self.lightgcn}\n"
            f")"
        )
