"""
Deduplicated semantic item features.

GUARDRAIL: transformer inference runs once per *unique* item string (hundreds to a few
thousand rows), never per transaction line. Customer-level features are then produced
with Polars joins + weighted group-bys.

Steps
-----
1. Unique texts: ``ItemName | ItemCategoryLevel2 | ItemCategoryLevel3`` over in-scope,
   non-adjustment lines.
2. Encode with sentence-transformers (``device="cuda"`` when available) and L2-normalise.
   If the model cannot be loaded (offline box, no weights), fall back to TF-IDF char
   n-grams so the pipeline still runs - logged loudly, and recorded in the cache key.
3. TruncatedSVD -> ``n_components`` (random_state=42) for compact taste dimensions.
4. Cosine similarity of every item with descriptions of the 17 authoritative
   Opportunity classes (full-dimensional, before SVD).
5. Per cutoff, customer aggregation over history ``ts < cutoff``: spend-weighted mean
   and recency-weighted (exponential, half-life ``half_life_days``) mean of all item
   dimensions, plus "novel affinity": similarity mass of the customer's recent quarter
   toward each class.

Leakage note: the encoder and SVD are unsupervised functions of item *strings*; no
targets are touched. Customer aggregations for cohort ``t`` only read rows ``< t``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD

from src.features import included_categories, safe_name
from src.utils import resolve_device, setup_logger

logger = setup_logger("sasol_embeddings")

EMB_VERSION = "e2"
ID = "ID"

# Descriptions for the 17 authoritative Opportunity classes (names come from label_config).
_CATEGORY_TEXT = {
    "Beverages": "cold drinks soft drinks soda juice bottled water energy drinks",
    "Food & beverage": "hot food prepared meals pies sandwiches takeaway coffee shop food",
    "Fuel": "fuel petrol diesel unleaded litres at the pump",
    "Other": "general merchandise car care car wash automotive coffee frozen food perishables miscellaneous",
    "Braai & Ice": "braai charcoal firewood firelighters bag of ice",
    "Breads & rolls": "bread loaves rolls buns bakery",
    "Confectionary": "chocolate sweets candy gum mints",
    "Groceries": "groceries milk sugar cooking oil pantry staples tinned food",
    "Ice-cream": "ice cream frozen dessert cones lollies",
    "Lubricants": "engine oil motor oil lubricants brake fluid coolant",
    "Snacking": "snacks crisps chips biltong nuts dried fruit",
    "Tobacco": "cigarettes tobacco cigars lighters",
}


def opportunity_descriptions(labels: Sequence[str]) -> Dict[str, str]:
    out = {}
    for lab in labels:
        if lab == "Inactivity":
            out[lab] = "customer stops visiting no purchases inactive churned lapsed"
        elif lab == "Stable":
            out[lab] = "regular routine purchases same products steady habitual spend"
        else:
            kind, cat = lab.split(": ", 1)
            verb = "buying more of usual" if kind.startswith("Existing") else "trying new first purchase of"
            out[lab] = f"{verb} {cat}: {_CATEGORY_TEXT.get(cat, cat)}"
    return out


# --------------------------------------------------------------------------- encoders
def _encode_sentence_transformers(texts: List[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    torch.manual_seed(42)
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb.astype(np.float32)


def _encode_tfidf(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, lowercase=True)
    x = vec.fit_transform(texts)
    return normalize(x).toarray().astype(np.float32)


def encode_texts(texts: List[str], backend: str, model_name: str, batch_size: int = 256) -> tuple[np.ndarray, str]:
    """Returns (L2-normalised embeddings, backend actually used)."""
    if backend in ("auto", "sentence_transformers"):
        device = resolve_device()
        try:
            logger.info(f"Encoding {len(texts):,} unique texts with {model_name} on {device}")
            return _encode_sentence_transformers(texts, model_name, batch_size, device), f"st:{model_name}"
        except Exception as e:  # offline / missing weights
            if backend == "sentence_transformers":
                raise
            logger.warning(f"sentence-transformers unavailable ({type(e).__name__}: {e}); "
                           "FALLING BACK to TF-IDF char n-gram embeddings")
    logger.info(f"Encoding {len(texts):,} unique texts with TF-IDF char n-grams")
    return _encode_tfidf(texts), "tfidf"


# --------------------------------------------------------------------------- item table
def build_item_table(tx: pl.DataFrame, label_config: dict, emb_cfg: dict, seed: int = 42) -> tuple[pl.DataFrame, List[str], str]:
    cats = included_categories(label_config)
    items = (
        tx.filter(~pl.col("is_adjustment") & pl.col("ItemCategoryLevel2").cast(pl.String).is_in(cats))
        .select(
            "ItemName",
            pl.col("ItemCategoryLevel2").cast(pl.String).alias("_l2"),
            pl.col("ItemCategoryLevel3").cast(pl.String).alias("_l3"),
        )
        .unique()
        .with_columns(
            pl.concat_str([pl.col("ItemName").fill_null(""), pl.col("_l2").fill_null(""), pl.col("_l3").fill_null("")],
                          separator=" | ").str.to_lowercase().alias("text")
        )
        .sort("text")
    )
    texts = items["text"].unique(maintain_order=True).to_list()
    labels = label_config["opportunity_labels"]
    desc = opportunity_descriptions(labels)
    all_emb, used = encode_texts(texts + [desc[l] for l in labels], emb_cfg.get("backend", "auto"),
                                 emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                 emb_cfg.get("batch_size", 256))
    item_emb, cls_emb = all_emb[: len(texts)], all_emb[len(texts):]
    sims = item_emb @ cls_emb.T  # cosine: rows already unit-norm

    k = int(min(emb_cfg.get("n_components", 16), max(1, item_emb.shape[1] - 1), max(1, len(texts) - 1)))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    red = svd.fit_transform(item_emb).astype(np.float32)
    logger.info(f"SVD({k}) explained variance: {svd.explained_variance_ratio_.sum():.3f}")

    dims = [f"sem_svd{i:02d}" for i in range(k)] + [f"sem_aff__{safe_name(l)}" for l in labels]
    text_tbl = pl.DataFrame({"text": texts, **{d: v for d, v in zip(dims, np.hstack([red, sims]).T)}})
    item_tbl = items.join(text_tbl, on="text", how="left").select(
        "ItemName", "_l2", "_l3", *dims
    )
    return item_tbl, dims, used


def customer_semantic_features(tx: pl.DataFrame, item_tbl: pl.DataFrame, dims: List[str], cutoff: str,
                               half_life_days: float = 30.0) -> pl.DataFrame:
    from datetime import datetime
    t = datetime.strptime(cutoff, "%Y-%m-%d")
    h = (
        tx.lazy().filter((pl.col("ts") < t) & ~pl.col("is_adjustment"))
        .with_columns(pl.col("ItemCategoryLevel2").cast(pl.String).alias("_l2"),
                      pl.col("ItemCategoryLevel3").cast(pl.String).alias("_l3"))
        .join(item_tbl.lazy(), on=["ItemName", "_l2", "_l3"], how="inner")
        .with_columns(
            pl.col("amount").clip(lower_bound=0.0).alias("_w_spend"),
            (pl.lit(0.5) ** (((pl.lit(t) - pl.col("ts")).dt.total_seconds() / 86400.0) / half_life_days)).alias("_w_rec"),
        )
        .with_columns((pl.col("_w_spend") * pl.col("_w_rec")).alias("_w_both"))
    )
    aggs = [pl.col("_w_spend").sum().alias("_sw"), pl.col("_w_both").sum().alias("_bw"),
            pl.len().alias("sem_n_lines")]
    for d in dims:
        aggs.append((pl.col(d) * pl.col("_w_spend")).sum().alias(f"_s_{d}"))
        aggs.append((pl.col(d) * pl.col("_w_both")).sum().alias(f"_r_{d}"))
    g = h.group_by(ID).agg(aggs).collect()
    out = g.select(
        ID,
        *[(pl.col(f"_s_{d}") / (pl.col("_sw") + 1e-6)).alias(f"{d}__spend_w") for d in dims],
        *[(pl.col(f"_r_{d}") / (pl.col("_bw") + 1e-6)).alias(f"{d}__recency_w") for d in dims],
    )
    aff = [f"{d}__recency_w" for d in dims if d.startswith("sem_aff__")]
    if aff:
        out = out.with_columns(
            pl.concat_list(aff).list.arg_max().cast(pl.Float64).alias("sem_top_affinity_idx"),
            (pl.max_horizontal(aff) - pl.mean_horizontal(aff)).alias("sem_affinity_peak"),
        )
    return out.with_columns(pl.col(ID).cast(pl.String))


def semantic_features(tx: pl.DataFrame, cutoffs: Sequence[str], label_config: dict, emb_cfg: dict,
                      processed_dir: Path, fingerprint: str, force_recompute: bool = False,
                      seed: int = 42) -> Dict[str, pl.DataFrame]:
    """Cached per-cutoff customer semantic features (data/processed/semantic_features.parquet)."""
    key_src = f"{EMB_VERSION}|{fingerprint}|{sorted(emb_cfg.items())}|{seed}"
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    path = Path(processed_dir) / "semantic_features.parquet"
    wanted = sorted(set(cutoffs))
    if path.exists() and not force_recompute:
        cached = pl.read_parquet(path)
        if "_cache_key" in cached.columns and cached.height and cached["_cache_key"][0] == key \
                and set(wanted) <= set(cached["cohort_cutoff"].unique().to_list()):
            logger.info(f"Loaded cached semantic features from {path}")
            return {c: cached.filter(pl.col("cohort_cutoff") == c).drop("cohort_cutoff", "_cache_key") for c in wanted}

    item_tbl, dims, used = build_item_table(tx, label_config, emb_cfg, seed)
    if used == "tfidf" and emb_cfg.get("backend", "auto") == "auto":
        logger.warning("Semantic features were built with the TF-IDF fallback, not a transformer")
    item_tbl.write_parquet(Path(processed_dir) / "item_embeddings.parquet")
    hl = float(emb_cfg.get("half_life_days", 30.0))
    per = {c: customer_semantic_features(tx, item_tbl, dims, c, hl) for c in wanted}
    pl.concat([df.with_columns(pl.lit(c).alias("cohort_cutoff")) for c, df in per.items()], how="diagonal") \
        .with_columns(pl.lit(key).alias("_cache_key")).write_parquet(path)
    logger.info(f"Cached semantic features ({len(dims)} item dims, backend={used}) -> {path}")
    return per
