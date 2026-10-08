# Sasol Customer Value Recruitment Challenge

Production-grade, leak-free predictive architecture for the **Sasol Customer Value Recruitment Challenge**.

---

## 🎯 Competition Multi-Task Objective

1. **`CLV_fuel`**: 3-month forward fuel volume prediction:
   $$z_{\text{fuel}} = \frac{\ln(1 + y_{\text{fuel}})}{s_{\text{fuel}}} \quad \text{where } s_{\text{fuel}} = 2.352825$$
2. **`CLV_nonfuel`**: 3-month forward convenience store spend prediction (Two-Stage Hurdle):
   $$z_{\text{nonfuel}} = \frac{\ln(1 + y_{\text{nonfuel}})}{s_{\text{nonfuel}}} \quad \text{where } s_{\text{nonfuel}} = 2.420090$$
3. **`Opportunity`**: 17-class behavioral state transition classification (Growth, Adoption, Inactivity, Stable) conditioned on future expected monetary value and calibrated via Nelder-Mead F1 threshold optimization.

---

## 🏗️ Repository Architecture

```text
├── configs/
│   └── base_config.yaml                  # Model and pipeline hyperparameters
├── data/
│   ├── raw/                             # Raw transaction histories & rules
│   └── processed/                       # Cached out-of-time feature tables
├── notebooks/
│   └── sasol_grandmaster_colab.ipynb    # Full Colab GPU pipeline
├── src/
│   ├── features.py                      # Polars rolling multi-window feature factory
│   ├── hf_embeddings.py                 # Hugging Face spend-weighted customer taste vectors
│   ├── metrics.py                       # Composite competition evaluation suite
│   ├── pipeline.py                      # Multi-cohort coordinator & submission exporter
│   ├── postprocessing.py               # Nelder-Mead multi-class decision threshold optimizer
│   ├── utils.py                         # Seed setting, memory reducer, logging
│   └── models/
│       ├── base.py                      # Abstract model wrapper
│       ├── regressors.py                # LightGBM, CatBoost, XGBoost, Hurdle regressors
│       └── classifiers.py               # GBDT multi-class engines
├── pack_colab.py                        # Automated packaging for Colab offloading
├── run_eda.py                           # Baseline exploration & target distributions
├── train.py                             # Main entrypoint
├── README_COLAB.md                      # Detailed Google Colab runbook
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Local Execution
```bash
pip install -r requirements.txt
python train.py
```

### 2. Google Colab GPU Offloading
1. Open [Google Colab](https://colab.research.google.com/) and upload `notebooks/sasol_grandmaster_colab.ipynb`.
2. Generate the package bundle locally:
   ```bash
   python pack_colab.py
   ```
3. Upload `sasol_full_bundle.zip` to Colab and select **Runtime -> Run all**.
4. The notebook will automatically execute the entire pipeline and download `submission_colab_ensemble.csv`.
