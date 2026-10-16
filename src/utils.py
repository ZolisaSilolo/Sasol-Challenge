import logging
import os
import random
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Lock Python, NumPy and (if installed) PyTorch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """'cuda' when an NVIDIA GPU is usable, else 'cpu'. Override with SASOL_DEVICE=cpu|cuda."""
    forced = os.environ.get("SASOL_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=10)
            return "cuda"
        except (subprocess.SubprocessError, OSError):
            pass
    return "cpu"


def setup_logger(name: str = "sasol_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def load_config(config_path: str = "configs/base_config.yaml") -> dict:
    import yaml

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found at: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
