#!/usr/bin/env bash
# =============================================================================
# Back up / restore agent session logs and run artefacts to GCS, so a fresh
# Codespace (or any machine with gcloud) can resume where you left off.
#
#   scripts/session_sync.sh push     local -> $GCS_BUCKET_URI/sessions/
#   scripts/session_sync.sh pull     $GCS_BUCKET_URI/sessions/ -> local
#   scripts/session_sync.sh status   list what is in the bucket
#
# Both directions skip files whose destination copy is newer (never clobbers
# fresher work) and never delete anything. Credential files are never uploaded.
# Push refuses to run if the bucket grants public (allUsers) access.
# =============================================================================
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"

log()  { printf '\033[1;34m[sessions]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[sessions][error]\033[0m %s\n' "$*" >&2; exit 1; }

ENV_FILE="${GCP_ENV_FILE:-$ROOT_DIR/.env}"
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }
BUCKET="${GCS_BUCKET_URI:-gs://sasol-train-dataset}"; BUCKET="${BUCKET%/}"
DEST="$BUCKET/sessions"
command -v gcloud >/dev/null 2>&1 || die "gcloud not installed"

# Never leave this machine: OAuth tokens, account lists, lock files, caches.
EXCLUDE='(^|.*/)(oauth_creds\.json|google_accounts\.json|credentials.*\.json|.*\.lock|\.env|.*token.*)$|(^|.*/)__pycache__/.*'

# local_path|bucket_subpath  (dirs are synced recursively; missing locals are skipped)
PAIRS=(
  "$HOME/.kiro/sessions|kiro/sessions"
  "$HOME/.kiro/agents|kiro/agents"
  "$HOME/.gemini/history|gemini/history"
  "$HOME/.gemini/tmp|gemini/tmp"
  "$ROOT_DIR/agents/sasol_analyst/.adk|adk/sasol_analyst"
  "$ROOT_DIR/logs|repo/logs"
  "$ROOT_DIR/submissions|repo/submissions"
)

assert_private() {
  local members
  members="$(gcloud storage buckets get-iam-policy "$BUCKET" --format='value(bindings.members)' 2>/dev/null)" \
    || die "Cannot read IAM policy of $BUCKET"
  if grep -qE 'allUsers|allAuthenticatedUsers' <<<"$members"; then
    die "$BUCKET is publicly accessible - refusing to upload session logs"
  fi
}

push() {
  assert_private
  for pair in "${PAIRS[@]}"; do
    local src="${pair%%|*}" sub="${pair#*|}"
    [[ -d "$src" ]] || continue
    log "push $src -> $DEST/$sub"
    gcloud storage rsync "$src" "$DEST/$sub" --recursive --skip-if-dest-has-newer-mtime \
      --exclude="$EXCLUDE" --quiet
  done
  date -u +%Y-%m-%dT%H:%M:%SZ | gcloud storage cp - "$DEST/LAST_PUSH" --quiet
  log "Done. Restore anywhere with: make sessions-pull"
}

pull() {
  gcloud storage ls "$DEST/" >/dev/null 2>&1 || { log "Nothing in $DEST yet"; return 0; }
  for pair in "${PAIRS[@]}"; do
    local dst="${pair%%|*}" sub="${pair#*|}"
    gcloud storage ls "$DEST/$sub/" >/dev/null 2>&1 || continue
    mkdir -p "$dst"
    log "pull $DEST/$sub -> $dst"
    gcloud storage rsync "$DEST/$sub" "$dst" --recursive --skip-if-dest-has-newer-mtime --quiet
  done
  log "Restored (last push: $(gcloud storage cat "$DEST/LAST_PUSH" 2>/dev/null || echo unknown))"
}

case "${1:-}" in
  push) push ;;
  pull) pull ;;
  status) gcloud storage ls -l -r "$DEST/**" 2>/dev/null | tail -n 25 ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
