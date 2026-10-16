"""Training entrypoint (local CPU or GCE GPU VM)."""

import argparse
import json

from src.pipeline import run_pipeline
from src.utils import setup_logger

logger = setup_logger("sasol_train")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sasol CLV multi-cohort 3-head training")
    ap.add_argument("--config", default="configs/base_config.yaml")
    ap.add_argument("--force-recompute", action="store_true", help="Rebuild labels/features/embeddings caches")
    ap.add_argument("--sub-name", default=None, help="Submission filename inside paths.sub_dir")
    ap.add_argument("--metrics-name", default=None, help="Metrics JSON filename inside paths.sub_dir")
    ap.add_argument("--no-oot", action="store_true", help="Skip the out-of-time benchmark (final fit only)")
    ap.add_argument("--no-embeddings", action="store_true", help="Disable semantic item embeddings")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="a.b.c=JSON",
                    help="Config override, e.g. --set validation.n_splits=3")
    args = ap.parse_args()

    overrides: dict = {}
    if args.no_oot:
        overrides.setdefault("validation", {})["oot"] = False
    if args.no_embeddings:
        overrides.setdefault("embeddings", {})["enabled"] = False
    if args.seed is not None:
        overrides["seed"] = args.seed
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = overrides
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value

    run_pipeline(args.config, overrides, force_recompute=args.force_recompute,
                 sub_name=args.sub_name, metrics_name=args.metrics_name)


if __name__ == "__main__":
    main()
