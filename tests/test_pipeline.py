"""Cohorts/labels, CV leakage guarantees, model wrappers, submission contract, end-to-end run."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import purge_cohorts, resolve_cohort_cutoffs
from src.models.gbdt import MultiClassifier
from src.models.hurdle import HurdleModel
from src.pipeline import make_folds, run_cv, run_pipeline
from src.submission import validate_submission

os.environ.setdefault("SASOL_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- cohorts / labels
def test_labels_identical_to_official_generator(store, synth_raw, tmp_path):
    cut = "2025-06-01"
    ours = store.labels(cut)
    out = tmp_path / "official.csv"
    subprocess.run([sys.executable, str(synth_raw / "label_rules.py"), "--transactions", str(synth_raw / "train.csv"),
                    "--cutoff", cut, "--output", str(out)], check=True, capture_output=True)
    off = pd.read_csv(out, dtype={"ID": str})
    m = ours.merge(off, on="ID", suffixes=("", "_off"), validate="one_to_one")
    assert len(m) == len(off) == len(ours)
    np.testing.assert_allclose(m.CLV_fuel, m.CLV_fuel_off, rtol=1e-12)
    np.testing.assert_allclose(m.CLV_nonfuel, m.CLV_nonfuel_off, rtol=1e-12)
    assert (m.Opportunity == m.Opportunity_off).all()
    assert set(ours.Opportunity) <= set(store.label_config["opportunity_labels"])


def test_scaling_constants(store):
    n = store.label_config["normalization"]
    assert n["CLV_fuel"] == 2.352825386185113 and n["CLV_nonfuel"] == 2.4200904988927476
    lab = store.labels("2025-03-01")
    np.testing.assert_allclose(lab.CLV_fuel, np.log1p(lab.fuel_litres) / 2.352825386185113)
    np.testing.assert_allclose(lab.CLV_nonfuel, np.log1p(lab.nonfuel_rand) / 2.4200904988927476)


def test_cohort_resolution_and_purge(label_config):
    cuts = resolve_cohort_cutoffs({}, label_config)
    assert cuts == label_config["calibration_cutoffs"]
    with pytest.raises(ValueError):
        resolve_cohort_cutoffs({"label_cutoffs": ["2025-10-01"]}, label_config)  # outcome past 2025-12-01
    # OOT 2025-09-01: 2025-06-01 cohort's outcome ends exactly at 2025-09-01 -> allowed.
    assert purge_cohorts(cuts[:-1], "2025-09-01") == cuts[:-1]
    assert purge_cohorts(["2025-07-01", "2025-06-01"], "2025-09-01") == ["2025-06-01"]


def test_group_folds_never_split_a_customer():
    ids = np.repeat(np.arange(200).astype(str), 5)
    folds = make_folds(ids, 5, 42)
    per_id = pd.Series(folds).groupby(ids).nunique()
    assert (per_id == 1).all() and set(folds) == set(range(5))


# ----------------------------------------------------------------------------- OOF leakage
class _Memoriser:
    """Returns the training target for rows it has seen, -1 otherwise: exposes any OOF leak."""

    def fit(self, X, y, w=None):
        self.seen = {r.tobytes(): v for r, v in zip(X, y)}
        return self

    def predict(self, X):
        return np.array([self.seen.get(r.tobytes(), -1.0) for r in X])


def test_run_cv_oof_never_sees_own_rows():
    X = np.arange(300, dtype=np.float32).reshape(-1, 1)
    y = np.random.default_rng(0).normal(size=300)
    folds = np.arange(300) % 5
    res = run_cv("leak", _Memoriser, X, y, None, folds, {"other": X[:10]}, lambda m, X: m.predict(X),
                 lambda a, b: 0.0, "x")
    assert (res.oof[:, 0] == -1.0).all()
    # Scoring rows get fold-averaged predictions (seen by 4 of 5 folds here).
    np.testing.assert_allclose(res.others["other"][:, 0], (4 * y[:10] - 1) / 5)


# ----------------------------------------------------------------------------- models
def test_multiclassifier_expands_missing_classes():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 4)), rng.choice([0, 3, 7], size=120)
    for eng in ("lgbm", "xgboost", "catboost"):
        p = MultiClassifier(eng, 17, {"n_estimators": 10} if eng != "catboost" else {"iterations": 10}).fit(X, y).predict_proba(X)
        assert p.shape == (120, 17)
        np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-5)
        assert p[:, [1, 2, 4, 5, 6] + list(range(8, 17))].max() == 0.0


def test_hurdle_outputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = np.where(X[:, 0] > 0, np.abs(X[:, 1]) + 1, 0.0)
    m = HurdleModel(["lgbm"], ["lgbm"], {"lgbm": {"n_estimators": 30}}, {"lgbm": {"n_estimators": 30}}).fit(X, y)
    out = m.predict(X)
    assert out.shape == (400, 3)
    assert (out[:, 1] >= 0).all() and (out[:, 1] <= 1).all() and (out[:, 0] >= 0).all()
    np.testing.assert_allclose(out[:, 0], out[:, 1] * out[:, 2])


# ----------------------------------------------------------------------------- submission contract
@pytest.fixture
def classes(label_config):
    return label_config["opportunity_labels"]


def _sub(n=5, classes=("Stable",)):
    return pd.DataFrame({"ID": [str(i) for i in range(n)], "CLV_fuel": 1.0, "CLV_nonfuel": 0.5,
                         "Opportunity": classes[0]})


def test_validate_submission_accepts_good(classes):
    s = _sub()
    validate_submission(s, s[["ID"]], classes, expected_rows=5)


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.iloc[:-1], "Expected"),
    (lambda s: s[["ID", "CLV_nonfuel", "CLV_fuel", "Opportunity"]], "columns"),
    (lambda s: s.iloc[::-1].reset_index(drop=True), "ordering"),
    (lambda s: s.assign(CLV_fuel=[np.nan, 1, 1, 1, 1]), "NaN"),
    (lambda s: s.assign(CLV_nonfuel=-0.1), "Negative"),
    (lambda s: s.assign(Opportunity="Growth"), "Invalid Opportunity"),
])
def test_validate_submission_rejects_bad(classes, mutate, msg):
    s = _sub()
    with pytest.raises(AssertionError, match=msg):
        validate_submission(mutate(s.copy()), s[["ID"]], classes, expected_rows=5)


@pytest.mark.skipif(not (REPO / "data/raw/SampleSubmission.csv").exists(),
                    reason="organiser SampleSubmission.csv not in data/raw/")
def test_real_sample_submission_shape(classes):
    sample = pd.read_csv(REPO / "data/raw/SampleSubmission.csv", dtype={"ID": str})
    assert sample.shape == (5488, 4)
    validate_submission(sample, sample, classes)  # expected_rows defaults to 5488


# ----------------------------------------------------------------------------- end-to-end
TINY = {
    "validation": {"n_splits": 3, "oot": True},
    "embeddings": {"enabled": True, "backend": "tfidf", "n_components": 8},
    "heads": {
        "fuel": {"engines": {"lgbm": {"n_estimators": 40}, "catboost": {"iterations": 40}, "xgboost": {"n_estimators": 40}},
                 "aux_huber": {"engine": "lgbm", "params": {"n_estimators": 40}}},
        "nonfuel": {"engines": {"lgbm": {"n_estimators": 40}, "catboost": {"iterations": 40}, "xgboost": {"n_estimators": 40}},
                    "hurdle": {"classifier_engines": ["lgbm"], "regressor_engines": ["lgbm"],
                               "classifier_params": {"lgbm": {"n_estimators": 40}},
                               "regressor_params": {"lgbm": {"n_estimators": 40}}}},
        "opportunity": {"engines": {"lgbm": {"n_estimators": 30}, "catboost": {"iterations": 30},
                                    "xgboost": {"n_estimators": 30}}},
    },
    "postprocessing": {"class_weights": {"grid_size": 7, "max_passes": 1}},
}


def test_end_to_end_pipeline(synth_raw, tmp_path):
    sample = pd.read_csv(synth_raw / "SampleSubmission.csv", dtype={"ID": str})
    over = {**TINY, "paths": {"raw_dir": str(synth_raw), "processed_dir": str(tmp_path / "proc"),
                              "sub_dir": str(tmp_path / "subs")},
            "submission": {"expected_rows": len(sample)}}
    report, sub = run_pipeline(str(REPO / "configs/fast_config.yaml"), over)

    assert sub.shape == (len(sample), 4)
    assert list(sub.columns) == ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]
    assert (sub.ID.values == sample.ID.values).all()
    assert (tmp_path / "subs" / "submission.csv").exists() and (tmp_path / "subs" / "oof_metrics.json").exists()
    assert (tmp_path / "proc" / "semantic_features.parquet").exists()

    oot = report["oot"]
    assert oot["cutoff"] == "2025-09-01" and "2025-09-01" not in oot["train_cutoffs"]
    for block in (oot["metrics"], report["final"]["oof"]):
        assert 0 <= block["f1_opportunity"] <= 1 and block["rmse_fuel"] >= 0 and block["rmse_nonfuel"] >= 0
    fin = report["final"]
    assert set(fin["fuel"]["blend_weights"]) == {"lgbm", "catboost", "xgboost", "huber_lgbm"}
    assert "hurdle" in fin["nonfuel"]["blend_weights"]
    assert set(fin["opportunity"]["blend_weights"]) == {"lgbm", "catboost", "xgboost"}

    # Determinism under the fixed seed: a second run (cached features) reproduces the submission.
    _, sub2 = run_pipeline(str(REPO / "configs/fast_config.yaml"), {**over, "validation": {"n_splits": 3, "oot": False}})
    pd.testing.assert_frame_equal(sub, sub2)


# ----------------------------------------------------------------------------- GCP orchestration (dry run)
@pytest.mark.parametrize("cmd,expect", [
    ("init", ["services enable compute.googleapis.com storage.googleapis.com",
              "storage buckets create gs://sasol-clv-dry-run-project", "storage rsync", "/data/raw",
              "networks create sasol-net", "--source-ranges 35.235.240.0/20", "--allow tcp:22"]),
    ("up", ["--image-family pytorch-latest-gpu", "--image-project deeplearning-platform-release",
            "--maintenance-policy TERMINATE", "--boot-disk-size 100GB", "--machine-type g2-standard-8",
            "remote_bootstrap.sh", "--network sasol-net", "--tunnel-through-iap"]),
    ("train", ["remote_train.sh launch", "storage cp"]),
    ("down", ["compute instances stop sasol-gpu-trainer"]),
    ("up:cpu", ["--machine-type c2d-highcpu-32", "--image-family debian-12", "--image-project debian-cloud",
                "--provisioning-model SPOT", "SASOL_MODE=cpu"]),
])
def test_gcp_cluster_dry_run(cmd, expect, synth_raw, tmp_path):
    cmd, _, mode = cmd.partition(":")
    # Drop any GCP settings exported by the developer's shell so the script's defaults are tested.
    base = {k: v for k, v in os.environ.items() if not k.startswith(("GCE_", "GCP_", "GCS_"))}
    env = {**base, "DRY_RUN": "1", "GCE_MODE": mode or "gpu", "GCP_PROJECT_ID": "dry-run-project", "GCS_BUCKET_URI": "",
           "GCP_ENV_FILE": str(tmp_path / "absent.env"), "RAW_DIR": str(synth_raw), "RUN_ID": "T1"}
    r = subprocess.run(["bash", "scripts/gcp_cluster.sh", cmd], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    plan = r.stderr.replace("\\ ", " ")
    for e in expect:
        assert e in plan, f"{e!r} missing from dry-run plan:\n{plan}"
    if mode == "cpu":
        assert "--accelerator" not in plan and "install-nvidia-driver" not in plan


def test_env_example_is_sourceable():
    r = subprocess.run(["bash", "-c", "set -euo pipefail; set -a; source .env.example; echo \"$GCE_FALLBACK_ZONES\""],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "", r.stderr
    assert r.stdout.split() == ["us-central1-b", "us-central1-c", "us-central1-f"]
