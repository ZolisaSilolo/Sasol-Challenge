#!/usr/bin/env bash
# Runs ON the GCE VM.
#
#   remote_train.sh launch <RUN_ID> <GCS_BUCKET_URI> [train.py args...]
#       Start training detached (survives SSH drops) and stream its log.
#       Exits with the training exit status.
#   remote_train.sh logs
#       Re-attach to the most recent training log.
#   remote_train.sh run <RUN_ID> <GCS_BUCKET_URI> [train.py args...]
#       Foreground job body (used by `launch`).
set -euo pipefail
cd "$HOME/sasol"
mkdir -p logs data/raw data/processed submissions

mode="${1:?mode required: launch|logs|status|run}"; shift

stream_until_done() {  # pid logfile
  local pid="$1" logf="$2"
  tail --pid="$pid" -n +1 -f "$logf" || true
  local rc; rc="$(cat logs/last.exit 2>/dev/null || echo 1)"
  echo "[remote] Training finished with exit status $rc"
  return "$rc"
}

case "$mode" in
  launch)
    run_id="${1:?RUN_ID required}"; bucket="${2:?GCS bucket required}"; shift 2
    if [[ -f logs/last.pid ]] && kill -0 "$(cat logs/last.pid)" 2>/dev/null; then
      echo "[remote] A training run is already active (pid $(cat logs/last.pid)). Use: make gcp-logs"; exit 1
    fi
    logf="logs/train_${run_id}.log"
    rm -f logs/last.exit
    # setsid+nohup: the job is detached from the SSH session's process group.
    setsid nohup bash -c 'bash scripts/remote_train.sh run "$@"; echo $? > logs/last.exit' _ \
      "$run_id" "$bucket" "$@" > "$logf" 2>&1 < /dev/null &
    echo $! > logs/last.pid
    ln -sf "$(basename "$logf")" logs/latest.log
    echo "[remote] Started run $run_id (pid $(cat logs/last.pid)) -> ~/sasol/$logf"
    stream_until_done "$(cat logs/last.pid)" "$logf"
    ;;

  status)   # RUNNING | EXIT:<code> | NONE
    if [[ -f logs/last.pid ]] && kill -0 "$(cat logs/last.pid)" 2>/dev/null; then echo RUNNING
    elif [[ -f logs/last.exit ]]; then echo "EXIT:$(cat logs/last.exit)"
    else echo NONE; fi
    ;;

  logs)
    [[ -f logs/last.pid ]] || { echo "[remote] No runs yet"; exit 1; }
    stream_until_done "$(cat logs/last.pid)" logs/latest.log
    ;;

  run)
    run_id="${1:?RUN_ID required}"; bucket="${2:?GCS bucket required}"; shift 2
    bucket="${bucket%/}"
    # Always ship the run log to the bucket, also when training fails.
    trap 'gcloud storage cp "logs/train_${run_id}.log" "$bucket/sessions/vm-logs/" >/dev/null 2>&1 || true' EXIT
    export PYTHONUNBUFFERED=1
    # shellcheck disable=SC1091
    source .venv/bin/activate

    echo "[remote] Pulling raw data from $bucket/data/raw"
    gcloud storage rsync "$bucket/data/raw" data/raw
    # Re-use cached cohort/feature parquet from earlier runs (keyed by feature version).
    if gcloud storage ls "$bucket/processed/" >/dev/null 2>&1; then
      echo "[remote] Restoring processed cache from $bucket/processed"
      gcloud storage rsync "$bucket/processed" data/processed
    fi
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || true

    sub_name="submission_${run_id}.csv"
    python train.py --sub-name "$sub_name" --metrics-name "oof_metrics_${run_id}.json" "$@"

    echo "[remote] Uploading artefacts to $bucket"
    gcloud storage rsync data/processed "$bucket/processed"
    gcloud storage cp "submissions/$sub_name" "submissions/oof_metrics_${run_id}.json" \
      "$bucket/submissions/$run_id/"
    printf '%s' "$run_id" | gcloud storage cp - "$bucket/submissions/LATEST"
    echo "[remote] Done: $bucket/submissions/$run_id/$sub_name"
    ;;

  *) echo "Unknown mode '$mode'"; exit 2 ;;
esac
