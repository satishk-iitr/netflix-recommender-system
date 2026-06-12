# Netflix Prize Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Competition-Netflix%20Prize-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RMSE-0.8821-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MAP@10-0.1834-blue?style=for-the-badge" />
</p>

> **Competition**: Recommendation Systems for Personalized Content Discovery  
> **Dataset**: Netflix Prize Dataset (100M+ ratings, 480k users, 17k movies)  
> **Author**: Satish Kumar · 24113114

---

## 🏆 Results

| Model | RMSE ↓ | MAE ↓ | MAP@10 ↑ | NDCG@10 ↑ |
|---|---|---|---|---|
| SVD | 0.9102 | 0.7156 | 0.1423 | 0.2156 |
| SVD++ | 0.9048 | 0.7098 | 0.1489 | 0.2234 |
| NeuMF | 0.9034 | 0.7089 | 0.1587 | 0.2389 |
| LightGCN | 0.8978 | 0.7021 | 0.1712 | 0.2512 |
| Ensemble (Weighted) | 0.8856 | 0.6934 | 0.1798 | 0.2634 |
| **Ensemble (Stacking)** | **0.8821** | **0.6901** | **0.1834** | **0.2712** |

> MAP@10 relevance threshold: rating ≥ 3.5 (per competition specification)

---

## 📁 Project Structure

```
netflix-recommender/
├── configs/                    # Model hyperparameter configs (YAML)
│   ├── svd.yaml
│   ├── neumf.yaml
│   └── lightgcn.yaml
├── data/
│   ├── download_data.py        # Dataset download via kagglehub
│   ├── raw/                    # Netflix Prize raw text files (gitignored)
│   └── processed/              # Parquet files after preprocessing (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_svd_baseline.ipynb   # SVD model training and evaluation
│   ├── 03_neumf.ipynb          # Neural MF training and evaluation
│   ├── 04_lightgcn.ipynb       # LightGCN training and evaluation
│   ├── 05_ensemble_eval.ipynb  # Ensemble methods and final comparison
│   └── 06_recommendation_analysis.ipynb  # Top-K quality analysis
├── src/
│   ├── data/
│   │   ├── loader.py           # Netflix raw file parser (Polars-first)
│   │   ├── preprocessor.py     # ID encoding, temporal splitting, cold-start filter
│   │   └── dataset.py          # PyTorch datasets and interaction matrix
│   ├── models/
│   │   ├── svd_model.py        # SVD / SVD++ / NMF (scikit-surprise)
│   │   ├── neumf.py            # Neural Matrix Factorization (PyTorch)
│   │   ├── lightgcn.py         # Light Graph Convolutional Network (PyTorch)
│   │   └── ensemble.py         # Weighted and Stacking ensembles
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE, MAE, MAP@K, NDCG@K, Coverage, ...
│   │   └── evaluator.py        # Unified evaluation harness
│   ├── recommendation/
│   │   └── topk.py             # Top-K generator with analysis tools
│   └── utils/
│       ├── config.py           # Paths, hyperparameters, constants
│       └── visualization.py    # Seaborn/Plotly plotting utilities
├── scripts/
│   ├── preprocess.py           # CLI: raw data → processed parquet
│   ├── train_svd.py            # CLI: train SVD/SVD++/NMF
│   ├── train_neumf.py          # CLI: train NeuMF
│   ├── train_lightgcn.py       # CLI: train LightGCN
│   ├── evaluate_all.py         # CLI: evaluate and compare all models
│   └── generate_topk.py        # CLI: generate Top-K recommendations
├── tests/
│   ├── test_metrics.py         # Unit tests for evaluation metrics
│   └── test_data_pipeline.py   # Unit tests for data pipeline
├── results/
│   ├── model_comparison.json   # Full evaluation results
│   └── predictions/
│       └── sample_predictions.csv
├── report/
│   └── report.html             # Technical report (print to PDF)
├── presentation/
│   └── slides.html             # Presentation slides (print to PDF)
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/satishk-iitr/netflix-recommender-system.git
cd netflix-recommender
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python data/download_data.py
# Copy the downloaded files to data/raw/
```

Or manually from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place in `data/raw/`.

### 3. Preprocess

```bash
python scripts/preprocess.py \
    --data-dir data/raw \
    --output-dir data/processed \
    --split-method temporal \
    --min-user-ratings 5 \
    --min-movie-ratings 5
```

### 4. Train Models

```bash
# SVD (CPU-friendly, fastest)
python scripts/train_svd.py \
    --data-dir data/processed \
    --model-type svd \
    --output-dir results/models/svd

# NeuMF (GPU recommended)
python scripts/train_neumf.py \
    --data-dir data/processed \
    --device cuda \
    --epochs 20 \
    --output-dir results/models/neumf

# LightGCN (GPU recommended)
python scripts/train_lightgcn.py \
    --data-dir data/processed \
    --device cuda \
    --epochs 50 \
    --output-dir results/models/lightgcn
```

### 5. Evaluate

```bash
python scripts/evaluate_all.py \
    --data-dir data/processed \
    --models-dir results/models \
    --output-dir results/metrics
```

### 6. Generate Top-K Recommendations

```bash
python scripts/generate_topk.py \
    --model-path results/models/svd \
    --data-dir data/processed \
    --k 10 \
    --output-dir results/predictions
```

---

## 📓 Notebooks

Run notebooks in order for a complete walkthrough:

```bash
jupyter lab notebooks/
```

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Dataset exploration, distributions, sparsity, temporal trends |
| `02_svd_baseline.ipynb` | SVD training, grid search, evaluation, Top-K examples |
| `03_neumf.ipynb` | NeuMF architecture, training curves, embedding analysis |
| `04_lightgcn.ipynb` | Graph construction, BPR training, model comparison |
| `05_ensemble_eval.ipynb` | Weight optimization, stacking, final comparison table |
| `06_recommendation_analysis.ipynb` | Coverage, diversity, success/failure case studies |

---

## 📐 Evaluation Methodology

**Train/Validation/Test Split**: Temporal per-user split (70% / 10% / 20%).  
Each user's ratings are sorted chronologically, ensuring the model learns from past behavior to predict future preferences.

**RMSE**: Standard rating prediction accuracy metric.  
$$RMSE = \sqrt{\frac{1}{N}\sum_{(u,i) \in \mathcal{T}}(\hat{r}_{ui} - r_{ui})^2}$$

**MAP@10**: Mean Average Precision at K=10. An item is considered **relevant if its true rating ≥ 3.5** (per competition specification).  
$$MAP@K = \frac{1}{|U|} \sum_{u \in U} AP@K(u)$$

---

## 🔗 References

1. Netflix Prize — BellKor's Pragmatic Chaos solution (2009)
2. Koren, Y. (2008). *Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model.* KDD.
3. He, X. et al. (2017). *Neural Collaborative Filtering.
4. He, X. et al. (2020). *LightGCN: Simplifying and Powering GNN for Recommendation.* SIGIR.
5. Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking.* UAI.
