#!/usr/bin/env bash
# Runs ON the GCE VM. Creates ~/sasol/.venv (Python 3.12 via uv) and installs
# requirements-gcp.txt. Idempotent: skipped when the requirements hash is unchanged.
set -euo pipefail
cd "$HOME/sasol"

PY_VERSION="${PY_VERSION:-3.12}"

SASOL_MODE="${SASOL_MODE:-gpu}"
if [[ "$SASOL_MODE" == "gpu" ]]; then
  echo "[bootstrap] Waiting for NVIDIA driver (DLVM installs it on first boot)..."
  for i in $(seq 1 60); do
    if nvidia-smi >/dev/null 2>&1; then break; fi
    if (( i == 60 )); then echo "[bootstrap] WARNING: nvidia-smi unavailable - continuing (CPU fallback)"; fi
    sleep 10
  done
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
else
  echo "[bootstrap] CPU mode: $(nproc) vCPUs, $(free -g | awk '/Mem:/{print $2}') GB RAM"
fi

HASH="$(cat requirements-gcp.txt; echo "mode=$SASOL_MODE")"
HASH="$(sha256sum <<<"$HASH" | cut -d' ' -f1)"
if [[ -f .venv/.req_hash && "$(cat .venv/.req_hash)" == "$HASH" ]]; then
  echo "[bootstrap] Environment up to date (requirements hash $HASH)"
  exit 0
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] Installing uv"
  if ! python3 -m pip --version >/dev/null 2>&1; then   # plain Debian images ship without pip
    sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv
  fi
  python3 -m pip install --user --quiet uv 2>/dev/null \
    || python3 -m pip install --user --quiet --break-system-packages uv
fi

if [[ ! -x .venv/bin/python ]]; then
  if [[ "$SASOL_MODE" == "gpu" ]] && python3 -c "import torch" 2>/dev/null; then
    # Reuse the DLVM's CUDA-matched torch instead of downloading a second copy.
    uv venv --system-site-packages --python "$(command -v python3)" .venv
  else
    uv venv --python "$PY_VERSION" .venv
  fi
fi
uv pip install --python .venv/bin/python -r requirements-gcp.txt
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  if [[ "$SASOL_MODE" == "cpu" ]]; then   # CPU-only wheel: small, no CUDA
    uv pip install --python .venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
  else
    uv pip install --python .venv/bin/python "torch==2.14.1"
  fi
fi

.venv/bin/python - <<'EOF'
import torch, xgboost, catboost, lightgbm, polars
print(f"[bootstrap] torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"[bootstrap] xgboost {xgboost.__version__} | catboost {catboost.__version__} | "
      f"lightgbm {lightgbm.__version__} | polars {polars.__version__}")
EOF
echo "$HASH" > .venv/.req_hash
echo "[bootstrap] Done."
