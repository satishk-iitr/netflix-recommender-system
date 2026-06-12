"""
Netflix Recommendation System — Data Module
============================================
Provides data loading, preprocessing, and PyTorch dataset utilities
for the Netflix Prize dataset.

Classes:
    NetflixDataLoader   — Parse and load raw Netflix Prize text files.
    DataPreprocessor    — Encode IDs, split data, compute statistics.
    RatingDataset       — PyTorch Dataset for pointwise (user, item, rating) training.
    PairwiseDataset     — PyTorch Dataset for pairwise BPR training with negative sampling.
    InteractionMatrix   — Sparse user-item interaction matrix with graph utilities.

Functions:
    create_data_loaders — Build train/val DataLoader instances from DataFrames.
"""

from src.data.loader import NetflixDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import (
    RatingDataset,
    PairwiseDataset,
    InteractionMatrix,
    create_data_loaders,
)

__all__ = [
    "NetflixDataLoader",
    "DataPreprocessor",
    "RatingDataset",
    "PairwiseDataset",
    "InteractionMatrix",
    "create_data_loaders",
]
