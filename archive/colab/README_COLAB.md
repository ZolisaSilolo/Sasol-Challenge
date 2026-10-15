# 🚀 Google Colab Offloading Guide: Sasol Customer Value Recruitment Challenge

This guide explains how to offload the end-to-end multi-task training pipeline from your local machine to **Google Colab** (CPU or GPU).

---

## 📦 What Was Built

1. **`sasol_full_bundle.zip` (16.3 MB)**
   - Complete package including all code, configs, label generator scripts, and `data/raw/train.csv`.
   - Ready for one-click upload directly to Google Colab.

2. **`sasol_code_bundle.zip` (50 KB)**
   - Lightweight package containing code and notebooks only (if you already host `train.csv` on Google Drive).

3. **`notebooks/sasol_grandmaster_colab.ipynb`**
   - Standalone, self-contained Google Colab notebook.
   - Includes GPU diagnostic, automated dependency setup, multi-cohort pooling, Polars feature factory, multi-head ensembling (LightGBM + CatBoost + XGBoost + Hurdle), Nelder-Mead F1 threshold optimization, metric breakdowns, and automatic submission download.

4. **Modular Codebase (`src/`)**
   - `src/hf_embeddings.py`: Hugging Face `sentence-transformers` representation layer generating 12-dim customer spend-weighted taste vectors and 17-class semantic cosine category affinity scores.
   - `src/features.py`: Polars rolling multi-window aggregations (14d, 30d, 60d, 90d, 180d), spend velocities, inter-purchase intervals (IPI), commuting and pay-cycle cadence, and station loyalty index.
   - `src/models/`: LGBM, CatBoost, XGBoost regressors, Hurdle regressor, and multi-class classifiers.
   - `src/pipeline.py`: Supports multi-cohort historical pooling (`2024-12-01`, `2025-03-01`, `2025-06-01` training pool) with strict out-of-time validation (`2025-09-01`).

---

## ⚡ Option 1: Run via Interactive Notebook (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and select:
   ```text
   notebooks/sasol_grandmaster_colab.ipynb
   ```
3. In Colab's left sidebar (Files icon 📁), drag and drop:
   ```text
   sasol_full_bundle.zip
   ```
4. In the Colab menu, select **Runtime -> Run all** (or press `Ctrl+F9`).
5. The notebook will:
   - Unpack the archive into standard directories.
   - Install all necessary dependencies.
   - Extract multi-cohort targets and high-velocity Polars features.
   - Train Head 1 (Fuel CLV), Head 2 (Non-Fuel Hurdle), and Head 3 (Opportunity multi-class).
   - Optimize the F1 decision thresholds via Nelder-Mead.
   - Validate out-of-time performance and assert submission contracts.
   - Automatically trigger browser download of `submission_colab_ensemble.csv`.

---

## 💻 Option 2: Run via Colab Terminal / Bash Cell

If you prefer executing the pipeline as a background script on Colab:

1. In a Colab cell, run:
   ```bash
   !unzip -q sasol_full_bundle.zip
   !pip install -q -r requirements.txt
   !python -m src.pipeline
   ```
2. Your submission will be produced in:
   ```text
   submissions/submission_colab_ensemble.csv
   ```
3. Download it to your local machine:
   ```python
   from google.colab import files
   files.download("submissions/submission_colab_ensemble.csv")
   ```

---

## 🎯 Verification & Integrity Checklist

The pipeline automatically validates:
- `sub.shape == (5488, 4)`
- Columns: `['ID', 'CLV_fuel', 'CLV_nonfuel', 'Opportunity']`
- No missing values (`NaN` count is 0)
- Positivity constraints: `CLV_fuel >= 0` and `CLV_nonfuel >= 0`
- Opportunity classes strictly match the 17 authoritative strings from `label_config.json`.
