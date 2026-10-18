import numpy as np
import pandas as pd
import pytest

from src.category_meta import ADOPT, GROWTH, category_outcomes, fit_category_meta, select_targets
from src.features import included_categories
from src.pipeline import make_folds


@pytest.mark.parametrize("cutoff", ["2025-03-01", "2025-06-01"])
def test_category_outcomes_reproduce_official_labels(store, tx, label_config, cutoff):
    """Re-deriving the winner from per-category outcomes must give label_rules' label and increment."""
    out = category_outcomes(tx, cutoff, label_config).to_pandas().set_index("ID")
    ref = store.labels(cutoff).set_index("ID")
    ref.index = ref.index.astype(str)
    assert set(out.index) == set(ref.index)
    out = out.loc[ref.index]
    cats = included_categories(label_config)            # sorted -> argmax ties resolve alphabetically
    G = out[[f"g__{c}" for c in cats]].to_numpy()
    A = out[[f"a__{c}" for c in cats]].to_numpy()
    gm, am = G.max(1), A.max(1)
    mapping = label_config["category_mapping"]
    is_adopt = (am > 0) & (am >= gm)
    is_grow = (gm > 0) & ~is_adopt
    lab = np.where(is_adopt, [f"{ADOPT}: {mapping[cats[i]][ADOPT]}" for i in A.argmax(1)],
                   np.where(is_grow, [f"{GROWTH}: {mapping[cats[i]][GROWTH]}" for i in G.argmax(1)], "Stable"))
    active = ~ref.Opportunity.eq("Inactivity").to_numpy()
    np.testing.assert_array_equal(lab[active], ref.Opportunity.to_numpy()[active])
    inc = np.where(is_adopt, am, np.where(is_grow, gm, 0.0))
    np.testing.assert_allclose(inc[active], ref.winning_increment_rand.to_numpy()[active], atol=1e-6)
    assert (G[~active] == 0).all() and (A[~active] == 0).all()
    assert (lab != "Stable").sum() > 0


def test_fit_category_meta_shapes_and_oof(store, tx, label_config):
    cut = "2025-03-01"
    Y = category_outcomes(tx, cut, label_config).to_pandas()
    rng = np.random.default_rng(0)
    # Features = noisy copy of outcome flags (strong signal) + noise columns.
    X = np.hstack([(Y.filter(like="__").to_numpy() > 0) + rng.normal(0, 0.3, (len(Y), Y.shape[1] - 1)),
                   rng.normal(size=(len(Y), 5))]).astype(np.float32)
    folds = make_folds(Y.ID.to_numpy(), 3, 42)
    classes = list(label_config["opportunity_labels"])
    res = fit_category_meta(X, Y, folds, {"test": X[:7]}, label_config, classes,
                            params={"n_estimators": 20, "min_child_samples": 5}, min_pos=5, n_jobs=1)
    targets = select_targets(Y, label_config, 5)
    assert len(targets) > 0
    assert res.train.shape == (len(Y), len(res.names)) and res.others["test"].shape == (7, len(res.names))
    assert np.isfinite(res.train).all() and np.isfinite(res.others["test"]).all()
    p = res.train[:, :len(targets)]
    assert ((p >= 0) & (p <= 1)).all()
    assert np.median(list(res.report["oof_auc"].values())) > 0.8
