#!/usr/bin/env bash
# =============================================================================
# Sasol CLV - GCP lifecycle orchestrator (local VS Code side).
#
#   scripts/gcp_cluster.sh <command> [args]
#
# Commands:
#   env         Detect gcloud project/account and (re)write .env
#   init        Enable APIs, create GCS bucket, sync data/raw -> $GCS_BUCKET_URI/data/raw
#   up          Create (L4 -> T4 fallback) or start the GPU VM, push code, bootstrap deps
#   sync-code   Push src/, configs/, scripts/, train.py, label rules to the VM (no data)
#   train       Run training remotely (detached, log streamed), then download submission
#   logs        Re-attach to the remote training log
#   fetch       Download submission(s) for RUN_ID (default: latest) from GCS
#   down        Stop the VM (disk retained, GPU billing stops)
#   delete      Delete the VM (asks for confirmation; bucket is kept)
#   status      Show VM state, zone and machine type
#   ssh         Open an interactive SSH session
#
# Every mutating gcloud call goes through `run`, so DRY_RUN=1 prints the command
# plan without touching GCP (used by the test suite / CI).
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"
STATE_FILE="$ROOT_DIR/.gcp_state"   # caches the zone the VM actually landed in
ENV_FILE="${GCP_ENV_FILE:-$ROOT_DIR/.env}"

# ----------------------------------------------------------------------------- helpers
log()  { printf '\033[1;34m[gcp]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[gcp][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[gcp][error]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    # stderr, so dry-run output never pollutes captured return values like $(provision)
    printf '[dry-run] %s\n' "$(printf '%q ' "$@")" >&2
    return 0
  fi
  "$@"
}

load_env() {
  local pre_mode="${GCE_MODE:-}"   # command-line GCE_MODE=... beats .env
  if [[ -f "$ENV_FILE" ]]; then
    set -a; # shellcheck disable=SC1090
    source "$ENV_FILE"; set +a
  else
    warn "$ENV_FILE not found - using defaults (run: make gcp-env, or copy .env.example to .env)"
  fi
  GCP_REGION="${GCP_REGION:-us-central1}"
  GCP_ZONE="${GCP_ZONE:-us-central1-a}"
  GCE_INSTANCE_NAME="${GCE_INSTANCE_NAME:-sasol-gpu-trainer}"
  GCE_MACHINE_TYPE="${GCE_MACHINE_TYPE:-g2-standard-8}"
  GCE_ACCELERATOR="${GCE_ACCELERATOR:-type=nvidia-l4,count=1}"
  GCE_FALLBACK_MACHINE_TYPE="${GCE_FALLBACK_MACHINE_TYPE:-n1-standard-8}"
  GCE_FALLBACK_ACCELERATOR="${GCE_FALLBACK_ACCELERATOR:-type=nvidia-tesla-t4,count=1}"
  GCE_FALLBACK_ZONES="${GCE_FALLBACK_ZONES:-us-central1-b us-central1-c us-central1-f}"
  GCE_IMAGE_FAMILY="${GCE_IMAGE_FAMILY:-pytorch-latest-gpu}"
  GCE_IMAGE_PROJECT="${GCE_IMAGE_PROJECT:-deeplearning-platform-release}"
  GCE_BOOT_DISK_SIZE="${GCE_BOOT_DISK_SIZE:-100GB}"
  # Dedicated VPC; SSH is only reachable through IAP (35.235.240.0/20), never the open internet.
  GCE_NETWORK="${GCE_NETWORK:-sasol-net}"
  GCE_SUBNET_RANGE="${GCE_SUBNET_RANGE:-10.42.0.0/24}"
  # GCE_MODE=gpu: L4 -> T4 DLVM. GCE_MODE=cpu: high-core CPU VM (no GPU quota needed).
  [[ -n "$pre_mode" ]] && GCE_MODE="$pre_mode"
  GCE_MODE="${GCE_MODE:-gpu}"
  [[ "$GCE_MODE" == "gpu" || "$GCE_MODE" == "cpu" ]] || die "GCE_MODE must be gpu or cpu (got '$GCE_MODE')"
  GCE_CPU_MACHINE_TYPE="${GCE_CPU_MACHINE_TYPE:-c2d-highcpu-32}"
  GCE_CPU_FALLBACK_MACHINE_TYPE="${GCE_CPU_FALLBACK_MACHINE_TYPE:-n2-highcpu-32}"
  GCE_CPU_IMAGE_FAMILY="${GCE_CPU_IMAGE_FAMILY:-debian-12}"
  GCE_CPU_IMAGE_PROJECT="${GCE_CPU_IMAGE_PROJECT:-debian-cloud}"
  GCE_SPOT="${GCE_SPOT:-1}"   # CPU mode only: Spot pricing (~60-90% cheaper, preemptible)

  if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    if command -v gcloud >/dev/null 2>&1; then
      GCP_PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
    fi
    if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
      [[ "$DRY_RUN" == "1" ]] || die "GCP_PROJECT_ID not set in .env and no gcloud default project"
      GCP_PROJECT_ID="dry-run-project"
    fi
  fi
  GCS_BUCKET_URI="${GCS_BUCKET_URI:-gs://sasol-clv-${GCP_PROJECT_ID}}"
  GCS_BUCKET_URI="${GCS_BUCKET_URI%/}"
}

require_gcloud() {
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  local acct
  acct="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
  [[ -n "$acct" ]] || die "No active gcloud account. Run: gcloud auth login && gcloud auth application-default login"
  log "Account: $acct | Project: $GCP_PROJECT_ID"
}

gc() { gcloud --project "$GCP_PROJECT_ID" "$@"; }

# Zone the instance currently lives in (may differ from GCP_ZONE after fallback).
instance_zone() {
  if [[ "$DRY_RUN" == "1" ]]; then echo "$GCP_ZONE"; return; fi
  local z
  z="$(gc compute instances list --filter="name=($GCE_INSTANCE_NAME)" --format='value(zone.basename())' 2>/dev/null | head -n1)"
  if [[ -n "$z" ]]; then echo "$z" > "$STATE_FILE"; echo "$z"; return; fi
  echo ""
}

instance_status() {
  local zone="$1"
  if [[ "$DRY_RUN" == "1" || -z "$zone" ]]; then echo "NOT_FOUND"; return; fi
  gc compute instances describe "$GCE_INSTANCE_NAME" --zone "$zone" --format='value(status)' 2>/dev/null || echo "NOT_FOUND"
}

require_zone() {
  local z; z="$(instance_zone)"
  [[ -n "$z" ]] || die "Instance '$GCE_INSTANCE_NAME' not found. Run: make gcp-up"
  echo "$z"
}

SSH_FLAGS=(--tunnel-through-iap --quiet --ssh-flag=-o --ssh-flag=ServerAliveInterval=30 --ssh-flag=-o --ssh-flag=ServerAliveCountMax=20)

remote() {  # remote <zone> <command-string>
  local zone="$1"; shift
  run gcloud compute ssh "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" "${SSH_FLAGS[@]}" --command "$*"
}

# ----------------------------------------------------------------------------- env
cmd_env() {
  command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not installed - cannot detect project"
  log "gcloud auth list:";   gcloud auth list
  log "gcloud config list:"; gcloud config list
  local project; project="$(gcloud config get-value project 2>/dev/null || true)"
  [[ -n "$project" ]] || die "No default project. Run: gcloud config set project <PROJECT_ID>"
  sed -e "s|^GCP_PROJECT_ID=.*|GCP_PROJECT_ID=${project}|" \
      "$ROOT_DIR/.env.example" > "$ROOT_DIR/.env"
  log "Wrote .env for project '$project'"
}

# ----------------------------------------------------------------------------- init
RAW_FILES=(train.csv test.csv SampleSubmission.csv label_rules.py label_config.json DATA_DICTIONARY.md LABEL_RULES.md)

ensure_network() {
  if [[ "$DRY_RUN" == "1" ]] || ! gc compute networks describe "$GCE_NETWORK" >/dev/null 2>&1; then
    log "Creating VPC $GCE_NETWORK + subnet $GCE_SUBNET_RANGE in $GCP_REGION"
    run gcloud compute networks create "$GCE_NETWORK" --project "$GCP_PROJECT_ID" --subnet-mode custom
    run gcloud compute networks subnets create "$GCE_NETWORK-$GCP_REGION" --project "$GCP_PROJECT_ID" \
        --network "$GCE_NETWORK" --region "$GCP_REGION" --range "$GCE_SUBNET_RANGE"
  else
    log "VPC $GCE_NETWORK exists"
  fi
  if [[ "$DRY_RUN" == "1" ]] || ! gc compute firewall-rules describe "$GCE_NETWORK-allow-iap-ssh" >/dev/null 2>&1; then
    log "Firewall: SSH (tcp:22) only from Google IAP range 35.235.240.0/20"
    run gcloud compute firewall-rules create "$GCE_NETWORK-allow-iap-ssh" --project "$GCP_PROJECT_ID" \
        --network "$GCE_NETWORK" --direction INGRESS --allow tcp:22 \
        --source-ranges 35.235.240.0/20 --target-tags sasol-iap-ssh
  fi
}

cmd_init() {
  require_gcloud
  log "Enabling Compute Engine + Cloud Storage APIs"
  run gcloud services enable compute.googleapis.com storage.googleapis.com iap.googleapis.com --project "$GCP_PROJECT_ID"
  ensure_network

  if [[ "$DRY_RUN" != "1" ]] && gc storage buckets describe "$GCS_BUCKET_URI" >/dev/null 2>&1; then
    log "Bucket $GCS_BUCKET_URI already exists"
  else
    log "Creating bucket $GCS_BUCKET_URI ($GCP_REGION)"
    run gcloud storage buckets create "$GCS_BUCKET_URI" --project "$GCP_PROJECT_ID" \
        --location "$GCP_REGION" --uniform-bucket-level-access
  fi

  local raw="${RAW_DIR:-data/raw}" missing=()
  for f in "${RAW_FILES[@]}"; do [[ -f "$raw/$f" ]] || missing+=("$f"); done
  if (( ${#missing[@]} )); then
    die "Missing in $raw/: ${missing[*]} (download train.csv from the competition page)"
  fi
  log "Syncing $raw -> $GCS_BUCKET_URI/data/raw/ (unchanged files are skipped)"
  # rsync only uploads new/changed objects, so re-running init is cheap.
  run gcloud storage rsync "$raw" "$GCS_BUCKET_URI/data/raw" \
      --exclude='(^|.*/)__pycache__/.*|.*\.pyc$|^\.gitkeep$'
}

# ----------------------------------------------------------------------------- up
resolve_image_family() {
  if [[ "$GCE_MODE" == "cpu" ]]; then echo "$GCE_CPU_IMAGE_FAMILY"; return; fi
  if [[ "$DRY_RUN" == "1" ]]; then echo "$GCE_IMAGE_FAMILY"; return; fi
  if gcloud compute images describe-from-family "$GCE_IMAGE_FAMILY" --project "$GCE_IMAGE_PROJECT" >/dev/null 2>&1; then
    echo "$GCE_IMAGE_FAMILY"; return
  fi
  warn "Image family '$GCE_IMAGE_FAMILY' not found; selecting newest PyTorch CUDA DLVM family"
  local fam
  fam="$(gcloud compute images list --project "$GCE_IMAGE_PROJECT" --no-standard-images \
        --filter='family~^pytorch AND family~cu' --sort-by=~creationTimestamp --limit=1 --format='value(family)')"
  [[ -n "$fam" ]] || die "Could not resolve a PyTorch GPU image family in $GCE_IMAGE_PROJECT"
  echo "$fam"
}

# Machine families with GPUs bundled into the machine type reject/ignore --accelerator.
accelerator_flag() {
  local mt="$1" acc="$2"
  case "$mt" in g2-*|a2-*|a3-*) ;; *) printf -- '--accelerator=%s' "$acc" ;; esac
}

create_instance() {  # zone machine_type accelerator image_family
  local zone="$1" mt="$2" acc="$3" fam="$4" accflag
  accflag="$(accelerator_flag "$mt" "$acc")"
  local args=(compute instances create "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone"
    --machine-type "$mt" --boot-disk-size "$GCE_BOOT_DISK_SIZE" --boot-disk-type pd-balanced
    --scopes cloud-platform --labels "app=sasol-clv,mode=$GCE_MODE"
    --network "$GCE_NETWORK" --subnet "$GCE_NETWORK-$GCP_REGION" --tags sasol-iap-ssh)
  if [[ "$GCE_MODE" == "cpu" ]]; then
    args+=(--image-family "$fam" --image-project "$GCE_CPU_IMAGE_PROJECT")
    if [[ "$GCE_SPOT" == "1" ]]; then
      args+=(--provisioning-model SPOT --instance-termination-action STOP)
    fi
  else
    args+=(--image-family "$fam" --image-project "$GCE_IMAGE_PROJECT"
      --maintenance-policy TERMINATE --metadata install-nvidia-driver=True)
    [[ -n "$accflag" ]] && args+=("$accflag")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then run gcloud "${args[@]}"; return 0; fi
  local err
  if err="$(gcloud "${args[@]}" 2>&1)"; then echo "$err" >&2; return 0; fi
  echo "$err" >&2
  # Capacity / quota / availability failures are retryable with the next candidate.
  if grep -qiE 'RESOURCE_POOL_EXHAUSTED|quota|does not exist in zone|not available|not supported|unsupported|does not have enough resources|ZONE_RESOURCE' <<<"$err"; then
    return 2
  fi
  return 1
}

provision() {
  local fam; fam="$(resolve_image_family)"
  local zones=("$GCP_ZONE"); read -r -a extra <<<"$GCE_FALLBACK_ZONES"; zones+=("${extra[@]}")
  local configs=("$GCE_MACHINE_TYPE|$GCE_ACCELERATOR" "$GCE_FALLBACK_MACHINE_TYPE|$GCE_FALLBACK_ACCELERATOR")
  if [[ "$GCE_MODE" == "cpu" ]]; then
    configs=("$GCE_CPU_MACHINE_TYPE|none" "$GCE_CPU_FALLBACK_MACHINE_TYPE|none")
  fi
  for cfg in "${configs[@]}"; do
    local mt="${cfg%%|*}" acc="${cfg#*|}"
    for zone in "${zones[@]}"; do
      log "Creating $GCE_INSTANCE_NAME: $mt ($acc) in $zone, image family $fam"
      local rc=0; create_instance "$zone" "$mt" "$acc" "$fam" || rc=$?
      if (( rc == 0 )); then [[ "$DRY_RUN" == "1" ]] || echo "$zone" > "$STATE_FILE"; echo "$zone"; return 0; fi
      (( rc == 2 )) || die "Instance creation failed with a non-capacity error (see above)"
      warn "No capacity/quota for $mt in $zone - trying next candidate"
    done
  done
  [[ "$GCE_MODE" == "cpu" ]] && die "No CPU capacity/quota for $GCE_CPU_MACHINE_TYPE / $GCE_CPU_FALLBACK_MACHINE_TYPE in $GCP_REGION"
  die "All L4/T4 zone candidates exhausted (GPU quota is often 0 on new projects: check NVIDIA_L4_GPUS / NVIDIA_T4_GPUS / GPUS_ALL_REGIONS under IAM & Admin -> Quotas). Or set GCE_MODE=cpu in .env"
}

wait_for_ssh() {
  local zone="$1"
  [[ "$DRY_RUN" == "1" ]] && { remote "$zone" "true"; return; }
  log "Waiting for SSH on $GCE_INSTANCE_NAME..."
  for _ in $(seq 1 30); do
    if gcloud compute ssh "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" "${SSH_FLAGS[@]}" \
         --command true >/dev/null 2>&1; then return 0; fi
    sleep 10
  done
  die "VM did not become reachable over SSH"
}

cmd_up() {
  require_gcloud
  local zone status
  zone="$(instance_zone)"
  status="$(instance_status "$zone")"
  case "$status" in
    RUNNING)            log "Instance already RUNNING in $zone" ;;
    TERMINATED|STOPPED|SUSPENDED)
      log "Starting existing instance in $zone"
      run gcloud compute instances start "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" ;;
    STOPPING|PROVISIONING|STAGING)
      die "Instance is $status - retry in a minute" ;;
    *)                  zone="$(provision)" ;;
  esac
  wait_for_ssh "$zone"
  cmd_sync_code "$zone"
  log "Bootstrapping Python environment on VM (skipped if requirements unchanged)"
  remote "$zone" "SASOL_MODE=$GCE_MODE bash ~/sasol/scripts/remote_bootstrap.sh"
  log "VM ready. Next: make gcp-train"
}

# ----------------------------------------------------------------------------- sync-code
CODE_PATHS=(src configs scripts train.py requirements-gcp.txt
            data/raw/label_rules.py data/raw/label_config.json)

cmd_sync_code() {
  require_gcloud
  local zone="${1:-}"; [[ -n "$zone" ]] || zone="$(require_zone)"
  log "Pushing code (${CODE_PATHS[*]}) to $GCE_INSTANCE_NAME:~/sasol"
  if [[ "$DRY_RUN" == "1" ]]; then
    run tar czf - --exclude=__pycache__ "${CODE_PATHS[@]}" \| gcloud compute ssh "$GCE_INSTANCE_NAME" --zone "$zone" \
        --command "mkdir -p ~/sasol && tar xzf - -C ~/sasol"
    return
  fi
  # A tar stream over one SSH connection: a single round trip and only code is sent (data comes from GCS).
  tar czf - --exclude=__pycache__ --exclude='*.pyc' "${CODE_PATHS[@]}" \
    | gcloud compute ssh "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" "${SSH_FLAGS[@]}" \
        --command "mkdir -p ~/sasol && rm -rf ~/sasol/src && tar xzf - -C ~/sasol"
  log "Code synced."
}

# ----------------------------------------------------------------------------- train / logs / fetch
cmd_train() {
  require_gcloud
  local zone; zone="$(require_zone)"
  local status; status="$(instance_status "$zone")"
  [[ "$DRY_RUN" == "1" || "$status" == "RUNNING" ]] || die "Instance is $status. Run: make gcp-up"
  local run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  local train_args="${TRAIN_ARGS:-}"
  log "Run $run_id: launching remote training (Ctrl-C detaches; training keeps running)"
  # The remote script starts training under nohup and tails its log, so an SSH drop
  # does not kill the job. Its exit status mirrors the training exit status.
  remote "$zone" "bash ~/sasol/scripts/remote_train.sh launch '$run_id' '$GCS_BUCKET_URI' $train_args" || true
  if [[ "$DRY_RUN" != "1" ]]; then
    # The stream ends on completion OR on an SSH drop: re-attach until the job reports an exit code.
    local st="" tries=0
    while :; do
      st="$(gcloud compute ssh "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" "${SSH_FLAGS[@]}" \
            --command 'bash ~/sasol/scripts/remote_train.sh status' 2>/dev/null | tail -n1 || true)"
      case "$st" in
        EXIT:*) break ;;
        RUNNING) warn "Log stream dropped; training still running - re-attaching"; remote "$zone" "bash ~/sasol/scripts/remote_train.sh logs" || true ;;
        *) tries=$((tries + 1)); (( tries > 30 )) && die "Lost contact with VM (Spot preemption?). Status: $(instance_status "$zone")"
           warn "VM unreachable (attempt $tries/30), retrying in 20s"; sleep 20 ;;
      esac
    done
    [[ "$st" == "EXIT:0" ]] || die "Remote training failed ($st). Inspect: make gcp-logs / make gcp-ssh"
  fi
  RUN_ID="$run_id" cmd_fetch
}

cmd_logs() {
  require_gcloud
  local zone; zone="$(require_zone)"
  remote "$zone" "bash ~/sasol/scripts/remote_train.sh logs"
}

cmd_fetch() {
  require_gcloud
  mkdir -p submissions
  local run_id="${RUN_ID:-}"
  if [[ -z "$run_id" ]]; then
    [[ "$DRY_RUN" == "1" ]] && run_id="DRYRUN" || \
    run_id="$(gcloud storage cat "$GCS_BUCKET_URI/submissions/LATEST" 2>/dev/null || true)"
    [[ -n "$run_id" ]] || die "No LATEST marker in $GCS_BUCKET_URI/submissions/"
  fi
  log "Downloading $GCS_BUCKET_URI/submissions/$run_id/ -> submissions/"
  run gcloud storage cp "$GCS_BUCKET_URI/submissions/$run_id/*" submissions/
  if [[ "$DRY_RUN" != "1" ]]; then
    local sub="submissions/submission_${run_id}.csv"
    [[ -f "$sub" ]] || die "Submission $sub was not produced"
    log "Validating $sub locally"
    python -m src.submission "$sub"
  fi
}

# ----------------------------------------------------------------------------- down / delete / status / ssh
cmd_down() {
  require_gcloud
  local zone; zone="$(instance_zone)"
  if [[ -z "$zone" && "$DRY_RUN" != "1" ]]; then log "No instance found - nothing to stop"; return 0; fi
  local status; status="$(instance_status "$zone")"
  if [[ "$status" == "TERMINATED" || "$status" == "STOPPED" ]]; then log "Instance already stopped"; return 0; fi
  log "Stopping $GCE_INSTANCE_NAME in $zone (boot disk retained; GPU/vCPU billing stops)"
  run gcloud compute instances stop "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone"
}

cmd_delete() {
  require_gcloud
  local zone; zone="$(require_zone)"
  if [[ "${CONFIRM:-}" != "yes" ]]; then
    read -r -p "Delete VM '$GCE_INSTANCE_NAME' and its boot disk (irreversible)? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { log "Aborted"; return 0; }
  fi
  run gcloud compute instances delete "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" --quiet
  rm -f "$STATE_FILE"
}

cmd_status() {
  require_gcloud
  run gcloud compute instances list --project "$GCP_PROJECT_ID" --filter="name=($GCE_INSTANCE_NAME)" \
      --format='table(name,zone.basename(),machineType.basename(),status,guestAccelerators[0].acceleratorType.basename())'
}

cmd_ssh() {
  require_gcloud
  local zone; zone="$(require_zone)"
  run gcloud compute ssh "$GCE_INSTANCE_NAME" --project "$GCP_PROJECT_ID" --zone "$zone" --tunnel-through-iap
}

# ----------------------------------------------------------------------------- main
main() {
  local cmd="${1:-help}"; shift || true
  if [[ "$cmd" != "env" && "$cmd" != "help" ]]; then load_env; fi
  case "$cmd" in
    env)        cmd_env ;;
    init)       cmd_init ;;
    up)         cmd_up ;;
    sync-code)  cmd_sync_code "$@" ;;
    train)      cmd_train ;;
    logs)       cmd_logs ;;
    fetch)      cmd_fetch ;;
    down)       cmd_down ;;
    delete)     cmd_delete ;;
    status)     cmd_status ;;
    ssh)        cmd_ssh ;;
    help|-h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *) die "Unknown command '$cmd' (see: scripts/gcp_cluster.sh help)" ;;
  esac
}

main "$@"
