"""
Hugging Face Semantic Feature Extraction Engine for Sasol Customer Value Challenge.

Generates:
1. Customer Spend-Weighted Taste Vectors (dense semantic representations of purchased items)
2. Semantic Cosine Affinity Scores to all 17 Opportunity Candidate Categories
3. Graceful fallback if torch/sentence-transformers is absent in the environment
"""

import math
from datetime import datetime
import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD

# Authoritative opportunity descriptions for semantic cosine mapping
OPPORTUNITY_CATEGORY_DESCRIPTIONS = {
    "Beverages": "Beverages sodas cold drinks fruit juice energy drinks water iced tea",
    "Braai & Ice": "Braai charcoal briquettes firewood firelighters bag of ice ice cubes",
    "Breads & rolls": "Fresh bakery bread white brown rolls buns loaves baguettes",
    "Confectionary": "Confectionary chocolates sweets candy gums gummies mints chocolate bars",
    "Food & beverage": "Food and beverage prepared meals pies toasted sandwiches hot food takeaway lunch",
    "Fuel": "Fuel service station diesel turbo petrol unleaded 93 95 pump litres fill up",
    "Groceries": "Groceries pantry items milk sugar coffee tea flour cooking oil tinned food",
    "Ice-cream": "Ice cream frozen dairy treats cones popsicles ice lollies tubs",
    "Lubricants": "Motor lubricants engine oil motor oil 5W30 10W40 brake fluid coolant transmission",
    "Snacking": "Snacks potato crisps chips biltong droewors dried fruit roasted nuts peanuts pretzels",
    "Tobacco": "Tobacco cigarettes cigars smoking lighters matches vape rolling paper",
    "Automotive Products": "Automotive car care wiper blades fuses tyre shine screen wash polish",
    "Car Wash": "Car wash vehicle cleaning vacuum wash and dry exterior auto detailing",
    "Coffee": "Fresh brewed coffee espresso cappuccino latte americano hot coffee beverage",
    "Frozen food": "Frozen meals frozen vegetables burger patties frozen chicken fish",
    "Perishables": "Perishable goods fresh dairy yogurt cheese butter eggs",
    "Other": "General miscellaneous store goods convenience merchandise",
}


def extract_customer_semantic_features(
    raw_transactions_path: str,
    cutoff_date: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_components: int = 12,
    seed: int = 42
) -> pl.DataFrame:
    """
    Extracts semantic customer taste vectors & opportunity category affinities.
    Computes features strictly leak-free before cutoff_date.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        # Graceful fallback for lightweight local environments
        return pl.DataFrame({"ID": []})

    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    
    # Read transactions strictly prior to cutoff
    df = pl.read_csv(raw_transactions_path, infer_schema_length=0)
    df = df.with_columns([
        pl.coalesce([
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%Z", strict=False),
            pl.col("DateTimeZA").str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
        ]).alias("dt"),
        pl.col("TotalExclAmount").cast(pl.Float64, strict=False).fill_null(0.0).alias("amount"),
        pl.col("ItemName").fill_null("Unknown").alias("item_name"),
    ])
    
    hist_tx = df.filter(pl.col("dt") < cutoff_dt)
    
    # 1. Extract unique non-empty item names (typically 1,000 - 2,500 unique items)
    unique_items = (
        hist_tx.select("item_name")
        .filter(pl.col("item_name").is_not_null() & (pl.col("item_name") != "Unknown"))
        .unique()
        .to_series()
        .to_list()
    )
    
    if not unique_items:
        return pl.DataFrame({"ID": []})
        
    cleaned_items = [str(x).strip().upper() for x in unique_items]
    
    # 2. Encode unique items using Hugging Face model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    item_embs_384 = model.encode(
        cleaned_items,
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    
    # 3. Dimensionality reduction on item embeddings (TruncatedSVD)
    actual_components = min(n_components, len(cleaned_items) - 1, 384)
    svd = TruncatedSVD(n_components=actual_components, random_state=seed)
    item_embs_reduced = svd.fit_transform(item_embs_384)
    
    # 4. Map back to item DataFrame
    emb_cols = [f"sem_dim_{i}" for i in range(actual_components)]
    item_dict = {"item_name": unique_items}
    for i, col_name in enumerate(emb_cols):
        item_dict[col_name] = item_embs_reduced[:, i].astype(np.float32)
        
    item_emb_df = pl.DataFrame(item_dict)
    
    # 5. Join item embeddings with customer transaction log
    joined = hist_tx.join(item_emb_df, on="item_name", how="inner")
    
    # Clip amount to positive spend for weighting
    joined = joined.with_columns([
        pl.col("amount").clip(lower_bound=0.0).alias("pos_spend")
    ])
    
    # Aggregate spend-weighted sums per customer
    aggs = [pl.col("pos_spend").sum().alias("total_pos_spend")]
    for col_name in emb_cols:
        aggs.append(
            (pl.col(col_name) * pl.col("pos_spend")).sum().alias(f"{col_name}_sum")
        )
        
    cust_emb = joined.group_by("ID").agg(aggs)
    
    # Normalize by total positive spend -> centroid taste vector
    norm_exprs = [
        (pl.col(f"{col_name}_sum") / (pl.col("total_pos_spend") + 1e-4)).alias(f"hf_taste_{col_name}")
        for col_name in emb_cols
    ]
    
    # 6. Encode Opportunity Categories & Compute Customer Category Affinities
    cat_keys = list(OPPORTUNITY_CATEGORY_DESCRIPTIONS.keys())
    cat_descs = [OPPORTUNITY_CATEGORY_DESCRIPTIONS[k] for k in cat_keys]
    cat_embs_384 = model.encode(cat_descs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    cat_embs_reduced = svd.transform(cat_embs_384)  # shape: (17, actual_components)
    
    # Construct customer affinity expressions (dot product with normalized category vectors)
    affinity_exprs = []
    for cat_idx, cat_name in enumerate(cat_keys):
        cat_vec = cat_embs_reduced[cat_idx]  # shape: (actual_components,)
        norm_cat = np.linalg.norm(cat_vec) + 1e-5
        cat_unit = cat_vec / norm_cat
        
        # Linear combination of hf_taste dims
        dot_terms = [
            pl.col(f"hf_taste_{emb_cols[i]}") * float(cat_unit[i])
            for i in range(actual_components)
        ]
        safe_cat_name = cat_name.replace(" ", "_").replace("&", "and")
        affinity_exprs.append(
            pl.sum_horizontal(dot_terms).alias(f"hf_aff_{safe_cat_name}")
        )
        
    final_features = cust_emb.select([pl.col("ID")] + norm_exprs).with_columns(affinity_exprs)
    return final_features
