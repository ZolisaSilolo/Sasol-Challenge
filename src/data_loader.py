"""
Ground-truth cohort generation and the single shared transaction parse.

Design
------
* ``train.csv`` is parsed exactly once, by the authoritative ``label_rules.prepare``
  (pandas, exact Decimal handling). Labels for every cohort come from
  ``label_rules.raw_labels`` + ``label_rules.transform`` on that same frame, so our
  ground truth is byte-identical to the organisers' generator.
* The prepared frame is projected into a compact Polars table
  (``data/processed/transactions.parquet``) that the feature factory consumes. Feature
  code therefore never re-parses timestamps or amounts differently from the labels.
* Every artefact is keyed by a fingerprint of (train.csv, label_rules.py,
  label_config.json); changing any of them invalidates the cache automatically.

Temporal protocol
-----------------
``cohort cutoff t``: features use transactions with ``_time < t``; targets use
``[t, t + 3 calendar months)``. A cohort is only *labelable* if ``t + 3 months`` is
``<= public_history_end_exclusive``. For an OOT benchmark at cutoff ``v`` we *purge*
every training cohort whose outcome window overlaps ``[v, ...)`` i.e. keep only
``t + 3 months <= v``.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence

import pandas as pd
import polars as pl

from src.utils import setup_logger

logger = setup_logger("sasol_data")

ID_COL = "ID"
HORIZON_MONTHS = 3
LABEL_COLUMNS = [
    ID_COL, "fuel_litres", "nonfuel_rand", "Opportunity",
    "winning_source_category", "winning_increment_rand", "CLV_fuel", "CLV_nonfuel",
]
TX_COLUMNS = [
    ID_COL, "TransactionId", "LocationId", "LineItemType", "ItemName",
    "ItemCategoryLevel1", "ItemCategoryLevel2", "ItemCategoryLevel3",
]


# --------------------------------------------------------------------------- helpers
def load_label_rules(raw_dir: Path) -> ModuleType:
    path = Path(raw_dir) / "label_rules.py"
    if not path.exists():
        raise FileNotFoundError(f"label_rules.py not found at {path.resolve()}")
    spec = importlib.util.spec_from_file_location("sasol_label_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_label_config(raw_dir: Path) -> dict:
    path = Path(raw_dir) / "label_config.json"
    if not path.exists():
        raise FileNotFoundError(f"label_config.json not found at {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def add_months(cutoff: str, months: int) -> pd.Timestamp:
    return pd.Timestamp(cutoff) + pd.DateOffset(months=months)


def purge_cohorts(cutoffs: Sequence[str], benchmark_cutoff: str) -> List[str]:
    """Training cohorts whose 3-month outcome window ends on/before ``benchmark_cutoff``."""
    b = pd.Timestamp(benchmark_cutoff)
    return [c for c in cutoffs if add_months(c, HORIZON_MONTHS) <= b]


def resolve_cohort_cutoffs(cfg_cohorts: dict, label_config: dict) -> List[str]:
    """Explicit list from config, else the organisers' ``calibration_cutoffs``.

    ``extra_monthly_cutoffs`` (optional) adds intermediate month-start cohorts for data
    augmentation; they are validated against the released history like all others.
    """
    cutoffs = list(cfg_cohorts.get("label_cutoffs") or label_config["calibration_cutoffs"])
    cutoffs += list(cfg_cohorts.get("extra_monthly_cutoffs") or [])
    end = pd.Timestamp(label_config["public_history_end_exclusive"])
    out = sorted({pd.Timestamp(c).strftime("%Y-%m-%d") for c in cutoffs})
    for c in out:
        if add_months(c, HORIZON_MONTHS) > end:
            raise ValueError(f"Cohort {c}: outcome window extends past released history ({end.date()})")
    return out


# --------------------------------------------------------------------------- store
@dataclass
class CohortStore:
    """Lazily parses train.csv once and serves cached labels / Polars transactions."""

    raw_dir: Path
    processed_dir: Path
    force_recompute: bool = False
    _rules: Optional[ModuleType] = field(default=None, init=False, repr=False)
    _pdf: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _tx: Optional[pl.DataFrame] = field(default=None, init=False, repr=False)
    _fingerprint: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self.raw_dir / "train.csv"
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training transactions missing at '{self.train_path.resolve()}'. "
                "train.csv is not tracked in git - download it from the competition page "
                "into data/raw/ (or run `make gcp-init` after placing it)."
            )
        self.label_config = load_label_config(self.raw_dir)

    # ----------------------------------------------------------------- fingerprint
    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            h = hashlib.sha256()
            for p in (self.train_path, self.raw_dir / "label_rules.py", self.raw_dir / "label_config.json"):
                h.update(_sha256(p).encode())
            self._fingerprint = h.hexdigest()[:16]
        return self._fingerprint

    def cache_path(self, stem: str) -> Path:
        return self.processed_dir / f"{stem}__{self.fingerprint}.parquet"

    # ----------------------------------------------------------------- raw parse
    @property
    def rules(self) -> ModuleType:
        if self._rules is None:
            self._rules = load_label_rules(self.raw_dir)
        return self._rules

    def _prepared(self) -> pd.DataFrame:
        if self._pdf is None:
            logger.info(f"Parsing {self.train_path} with label_rules.load_transactions (single parse)...")
            self._pdf = self.rules.load_transactions(str(self.train_path))
            logger.info(f"Parsed {len(self._pdf):,} transaction lines")
        return self._pdf

    def release_raw(self) -> None:
        """Free the pandas frame once all labels are materialised."""
        self._pdf = None
        gc.collect()

    # ----------------------------------------------------------------- transactions
    def transactions(self) -> pl.DataFrame:
        if self._tx is not None:
            return self._tx
        path = self.cache_path("transactions")
        if path.exists() and not self.force_recompute:
            logger.info(f"Loading cached transactions from {path}")
            self._tx = pl.read_parquet(path)
            return self._tx

        d = self._prepared()
        adjustment = set(self.label_config["adjustment_items"])
        tx = pl.from_pandas(d[TX_COLUMNS + ["_time", "_quantity", "_amount"]]).with_columns(
            pl.col("_time").cast(pl.Datetime("us")).alias("ts"),
            (pl.col("_quantity").cast(pl.Float64) / 1e6).alias("qty"),
            (pl.col("_amount").cast(pl.Float64) / 1e6).alias("amount"),
            pl.col("ItemName").is_in(list(adjustment)).alias("is_adjustment"),
        ).drop("_time", "_quantity", "_amount")
        tx = tx.with_columns(
            [pl.col(c).cast(pl.Categorical) for c in
             ("LineItemType", "ItemCategoryLevel1", "ItemCategoryLevel2", "ItemCategoryLevel3", "LocationId")]
        ).sort([ID_COL, "ts"])
        tx.write_parquet(path)
        logger.info(f"Cached {tx.height:,} transactions -> {path}")
        self._tx = tx
        return tx

    # ----------------------------------------------------------------- labels
    def labels(self, cutoff: str) -> pd.DataFrame:
        """Authoritative labels for ``cutoff`` via label_rules (cached as parquet)."""
        if add_months(cutoff, HORIZON_MONTHS) > pd.Timestamp(self.label_config["public_history_end_exclusive"]):
            raise ValueError(f"Cutoff {cutoff} has no complete outcome window in the released history")
        path = self.cache_path(f"labels_{cutoff}")
        if path.exists() and not self.force_recompute:
            return pd.read_parquet(path)
        logger.info(f"Generating ground-truth labels for cohort {cutoff} via label_rules.raw_labels")
        raw = self.rules.raw_labels(self._prepared(), cutoff, self.label_config)
        z = self.rules.transform(raw, self.label_config)
        out = raw.merge(z[[ID_COL, "CLV_fuel", "CLV_nonfuel"]], on=ID_COL, how="left")[LABEL_COLUMNS]
        out[ID_COL] = out[ID_COL].astype(str)
        out.to_parquet(path, index=False)
        dist = out["Opportunity"].value_counts(normalize=True).head(4).round(3).to_dict()
        logger.info(f"Cohort {cutoff}: {len(out):,} eligible customers | top classes {dist}")
        return out

    def materialise(self, cutoffs: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """Build labels for all cutoffs + the Polars transaction table, then free pandas."""
        labels = {c: self.labels(c) for c in cutoffs}
        self.transactions()
        self.release_raw()
        return labels
