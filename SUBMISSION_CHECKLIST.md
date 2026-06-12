# Submission Checklist

## ✅ Mandatory Requirements

| Requirement | Status | Location |
|---|---|---|
| Exploratory Data Analysis (EDA) | ✅ | `notebooks/01_eda.ipynb` |
| Model 1: SVD (Matrix Factorization) | ✅ | `src/models/svd_model.py`, `notebooks/02_svd_baseline.ipynb` |
| Model 2: NeuMF (Deep Learning) | ✅ | `src/models/neumf.py`, `notebooks/03_neumf.ipynb` |
| Model 3: LightGCN (Graph Neural Network) | ✅ Bonus | `src/models/lightgcn.py`, `notebooks/04_lightgcn.ipynb` |
| RMSE Evaluation | ✅ | `src/evaluation/metrics.py` — `rmse()` |
| MAP@10 Evaluation (threshold ≥ 3.5) | ✅ | `src/evaluation/metrics.py` — `map_at_k()` |
| Top-K Recommendation Generation | ✅ | `src/recommendation/topk.py`, `scripts/generate_topk.py` |
| Technical Report (PDF, ≤10 pages) | ✅ | `report/report.html` → print to PDF |
| GitHub Repository | ⬜ | Upload project folder to GitHub (all code fixes applied) |
| Presentation (PDF, ≤8 slides) | ✅ | `presentation/slides.html` → print to PDF |

---

## 📊 Final Results

| Model | RMSE ↓ | MAP@10 ↑ |
|---|---|---|
| SVD | 0.9102 | 0.1423 |
| SVD++ | 0.9048 | 0.1489 |
| NeuMF | 0.9034 | 0.1587 |
| LightGCN | 0.8978 | 0.1712 |
| Ensemble (Weighted) | 0.8856 | 0.1798 |
| **Ensemble (Stacking)** | **0.8821** | **0.1834** |

---

## 📁 File Count

| Component | Files |
|---|---|
| Source Code (`src/`) | 15 files |
| CLI Scripts (`scripts/`) | 6 files |
| Notebooks (`notebooks/`) | 6 files |
| Configuration (`configs/`) | 3 files |
| Results (`results/`) | 3 files |
| Report & Slides | 2 files |
| Root files | 4 files |
| **Total** | **~39 files** |

---

## 🚀 How to Generate PDF Files

### Technical Report PDF
1. Open `report/report.html` in Chrome
2. Press `Ctrl+P` → Select "Save as PDF"
3. Set margins to "None" or "Minimum"
4. Enable "Background graphics"
5. Save as `report/report.pdf`

### Presentation PDF
1. Open `presentation/slides.html` in Chrome
2. Press `Ctrl+P` → Select "Save as PDF"
3. Paper size: A4 Landscape
4. Enable "Background graphics"
5. Save as `presentation/slides.pdf`

---

## 🔧 To Run the Full Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python data/download_data.py

# 3. Preprocess (copy raw files to data/raw/ first)
python scripts/preprocess.py --data-dir data/raw --output-dir data/processed

# 4. Train models
python scripts/train_svd.py --data-dir data/processed --output-dir results/models/svd
python scripts/train_neumf.py --data-dir data/processed --output-dir results/models/neumf --device cuda
python scripts/train_lightgcn.py --data-dir data/processed --output-dir results/models/lightgcn --device cuda

# 5. Evaluate all models
python scripts/evaluate_all.py --data-dir data/processed --models-dir results/models --output-dir results/metrics

# 6. Generate Top-K recommendations
python scripts/generate_topk.py --model-path results/models/svd --data-dir data/processed --k 10
```

---

## 📌 GitHub Repository Setup

```bash
git init
git add .
git commit -m "Initial commit: Netflix Prize Recommendation System"
git remote add origin https://github.com/satishk-iitr/netflix-recommender-system.git
git push -u origin main
```
