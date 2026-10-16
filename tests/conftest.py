"""Shared fixtures: a synthetic raw dir run through the *real* label_rules.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.synthetic import write_synthetic_raw_dir

REPO = Path(__file__).resolve().parents[1]
REAL_RAW = REPO / "data" / "raw"


@pytest.fixture(scope="session")
def synth_raw(tmp_path_factory) -> Path:
    return write_synthetic_raw_dir(tmp_path_factory.mktemp("raw"), REAL_RAW, n_customers=300, seed=42)


@pytest.fixture(scope="session")
def store(synth_raw, tmp_path_factory):
    from src.data_loader import CohortStore

    return CohortStore(synth_raw, tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="session")
def tx(store):
    return store.transactions()


@pytest.fixture(scope="session")
def label_config(store):
    return store.label_config
