# Sasol Customer Value Recruitment Challenge

A leak-free, multi-cohort, three-head GBDT stack that predicts 3-month fuel volume, non-fuel spend and a 17-class "Opportunity" label for each of 5,488 forecourt customers. It trains locally or on Google Cloud Spot VMs.

**Project write-up with architecture diagrams: [docs/WRITEUP.md](docs/WRITEUP.md)**

![Cloud architecture](docs/images/cloud_architecture.png)

> Data notice: the competition data and organiser files (`train.csv`, `test.csv`, `SampleSubmission.csv`, `label_rules.py`, `label_config.json`, `LABEL_RULES.md`, `DATA_DICTIONARY.md`) are Sasol's intellectual property and are not in this repository. Neither are submissions, logs or caches. To run the code or tests, place the organiser files in `data/raw/`.

Stack: Python 3.12, Polars, LightGBM, CatBoost, XGBoost, scikit-learn, scipy, sentence-transformers, Google Compute Engine, Cloud Storage, IAP, Google ADK + Vertex AI Gemini. No AutoML. Global seed 42.

## Status

The competition closed on 19 Oct 2026, and every cloud resource used by this project has since been deleted (VM, disk, bucket, VPC, firewall). The `gcp-*` targets still work against a new project; see [Google Cloud](#google-cloud).

## Targets and metric

| Target | Definition (next 3 calendar months) | Metric |
|---|---|---|
| `CLV_fuel` | `ln(1 + litres) / 2.352825386185113` | RMSE, weight 0.3 |
| `CLV_nonfuel` | `ln(1 + rand) / 2.4200904988927476` | RMSE, weight 0.3 |
| `Opportunity` | one of 17 classes from the organiser's label rules | weighted F1, weight 0.4 |

Internal loss: `composite = 0.3·RMSE_f + 0.3·RMSE_nf + 0.4·(1 − wF1)`, lower is better. The leaderboard scale, reverse-engineered from public scores, is `0.3·(1 − RMSE_f/0.74) + 0.3·(1 − RMSE_nf/0.816) + 0.4·wF1`, higher is better.

## Results

Out-of-time benchmark: train on cohorts whose outcome window closes on or before 2025-09-01 (overlapping cohorts purged), then score the 2025-09-01 cohort.

| Run (config) | Train rows | OOT RMSE fuel | OOT RMSE non-fuel | OOT wF1 | Composite ↓ | LB-scale ↑ |
|---|---|---|---|---|---|---|
| `fast_config` (5 quarterly cohorts) | 22,679 | 0.5995 | 0.7441 | 0.5093 | 0.5993 | 0.2871 |
| `fast_config` + f4 features | 22,679 | 0.5989 | 0.7438 | 0.5181 | 0.5956 | 0.2910 |
| **`gcp_tuned_config`** (17 monthly cohorts) | 70,678 | 0.5988 | 0.7451 | 0.5211 | **0.5948** | **0.2917** |
| `gcp_v2_config` (deeper trees) | 70,678 | 0.5992 | 0.7452 | 0.5149 | 0.5974 | 0.2891 |

Discussion, ablations and negative results are in [docs/WRITEUP.md](docs/WRITEUP.md#4-results).

## Method

- Cohorts: each cutoff `t` gives one row per customer. Features use rows with `ts < t`; targets use `[t, t + 3 months)`. The five organiser calibration cutoffs can be augmented with month-start cutoffs. Labels come from the organiser's `label_rules.py`, and a test asserts they are byte-identical.
- Validation: the OOT benchmark above, then a final fit with 5-fold `GroupKFold` on customer `ID` across all pooled cohorts.
- Head 1 (fuel): LGBM, CatBoost and XGBoost, plus a Huber LGBM, simplex-blended on OOF and clipped at 0.
- Head 2 (non-fuel): direct GBDTs plus a hurdle model, P(z > 0) × E[z | z > 0], blended on OOF.
- Head 3 (Opportunity): GBDTs on the base features plus OOF meta-features from Heads 1–2. Then nested-OOF class multipliers, `argmax_c(w_c · P_c)` with `w ∈ [0.2, 5]^17`, which fall back to argmax when they don't help.

![Modelling pipeline](docs/images/ml_pipeline.png)

## Layout

```text
train.py                     training entry point (config + --set overrides)
configs/
  base_config.yaml           full ensembles
  fast_config.yaml           small ensembles (laptop)
  fast_monthly_config.yaml   fast_config + 12 month-start augmentation cohorts
  gcp_tuned_config.yaml      best run: monthly cohorts + regularised trees
  gcp_v2_config.yaml         deeper-tree variant (slightly worse than tuned)
src/
  data_loader.py             one parse via label_rules.prepare, per-cohort labels, parquet cache
  features.py                Polars RFM / momentum / cadence / per-category / rule-replay features
  embeddings.py              sentence-transformer on unique item texts -> SVD(16) + class affinities
  category_meta.py           per-category label-rule meta-models (experimental)
  context_features.py        station context and fuel-grade features (experimental)
  survival_features.py       gap-survival churn features (experimental)
  models/gbdt.py             LGBM / CatBoost / XGBoost wrappers, auto CPU/GPU
  models/hurdle.py           hurdle regressor for CLV_nonfuel
  pipeline.py                cohorts, OOT benchmark, GroupKFold, three-head OOF stack
  postprocessing.py          simplex blending, nested-OOF class multipliers
  metrics.py                 RMSE / weighted F1 / composite
  submission.py              pre-flight submission assertions (also a CLI)
scripts/
  gcp_cluster.sh             GCP lifecycle (wrapped by the Makefile)
  remote_bootstrap.sh        on-VM uv venv setup (skipped when requirements are unchanged)
  remote_train.sh            on-VM detached training (setsid + nohup)
  remote_sweep.sh            on-VM hyper-parameter sweep using exp_harness.py
  session_sync.sh            back up / restore logs and sessions to GCS
  exp_harness.py             fast OOT harness for feature/data ideas
  opp_harness.py             Opportunity decision-layer experiments
  reg_harness.py             direct vs hurdle CLV experiments
agents/sasol_analyst/        Google ADK agent (Gemini on Vertex AI) with tool-grounded analysis
docs/                        write-up, diagrams (docs/diagrams/make_diagrams.py) and images
tests/                       pytest suite on synthetic data through the real label rules
data/raw/                    organiser files go here (not committed)
data/processed/              parquet caches keyed by a hash of train.csv + label rules
```

## Local

```bash
pip install -r requirements.txt
# put the organiser files in data/raw/
make test                                               # pytest on synthetic data
make train-fast                                         # fast_config -> submissions/
make train TRAIN_ARGS="--config configs/gcp_tuned_config.yaml"   # best config
python -m src.submission submissions/submission.csv     # re-validate any file
```

## Google Cloud

```bash
cp .env.example .env   # or: make gcp-env (detects gcloud account/project)
make gcp-init          # enable Compute + Storage APIs, create bucket, rsync data/raw
make gcp-run-cpu       # Spot CPU VM: up -> sync code -> train -> fetch + validate -> stop
make gcp-run           # same on an L4 GPU VM (T4 fallback) - needs GPU quota
```

Individual steps: `gcp-up`, `gcp-sync-code`, `gcp-train`, `gcp-logs`, `gcp-fetch`, `gcp-down`, `gcp-delete`, `gcp-status`. Prefix any target with `DRY_RUN=1` to print the gcloud plan without running it.

![Run lifecycle](docs/images/run_lifecycle.png)

Security: the VM sits in a dedicated VPC. SSH is only allowed from Google's IAP range (35.235.240.0/20), so no ports are open to the internet. The bucket uses uniform bucket-level access, and `session_sync.sh` refuses to push to a publicly readable bucket.

Quota: new projects usually have 0 GPUs and `CPUS_ALL_REGIONS = 12`. Set `GCE_CPU_MACHINE_TYPE` in `.env` to fit your quota; the original runs used `n2-custom-12-24576`. At about 5k customers per cohort, CPU training is enough.

## Submission contract

`src/submission.py` checks every file before it is written, and again after it is downloaded from GCS:

- shape is (5488, 4)
- columns are `ID, CLV_fuel, CLV_nonfuel, Opportunity`
- ID order matches `SampleSubmission.csv`
- no NaN values
- CLV predictions ≥ 0
- every Opportunity label is one of the 17 classes
