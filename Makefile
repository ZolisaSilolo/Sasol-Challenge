# Sasol CLV - local + Google Cloud lifecycle.
# All gcp-* targets read .env (see .env.example). Prefix with DRY_RUN=1 to print the
# gcloud command plan without executing it.
SHELL := /bin/bash
PYTHON ?= python
GCP := scripts/gcp_cluster.sh
# gcloud is often installed user-local (e.g. Codespaces: ~/.local/bin).
export PATH := $(HOME)/.local/bin:$(HOME)/google-cloud-sdk/bin:$(PATH)

.PHONY: help install test train train-fast gcp-env gcp-init gcp-up gcp-sync-code \
        gcp-train gcp-logs gcp-fetch gcp-down gcp-delete gcp-status gcp-ssh gcp-run gcp-run-cpu sessions-push sessions-pull sessions-status

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n",$$1,$$2}'

# ----------------------------------------------------------------- local
install: ## Install pinned local dependencies
	$(PYTHON) -m pip install -r requirements.txt

test: ## Run unit + end-to-end tests on synthetic data
	$(PYTHON) -m pytest -q tests

train: ## Train locally (CPU fallback) with the default config
	$(PYTHON) train.py $(TRAIN_ARGS)

train-fast: ## Quick local smoke run (no embeddings, small ensembles)
	$(PYTHON) train.py --config configs/fast_config.yaml $(TRAIN_ARGS)

# ----------------------------------------------------------------- GCP
gcp-env: ## Detect gcloud project/account and write .env
	@bash $(GCP) env

gcp-init: ## Enable APIs, create bucket, sync data/raw to GCS
	@bash $(GCP) init

gcp-up: ## Create/start GPU VM (L4, T4 fallback), push code, install deps
	@bash $(GCP) up

gcp-sync-code: ## Push src/, configs/, scripts/ to the VM (no data)
	@bash $(GCP) sync-code

gcp-train: gcp-sync-code ## Train on the VM, stream logs, download validated submission
	@bash $(GCP) train

gcp-logs: ## Re-attach to the remote training log
	@bash $(GCP) logs

gcp-fetch: ## Download submission for RUN_ID (default: latest) from GCS
	@bash $(GCP) fetch

gcp-down: ## Stop the VM (stops GPU billing; disk is kept)
	@bash $(GCP) down

gcp-delete: ## Delete the VM (asks for confirmation)
	@bash $(GCP) delete

gcp-status: ## Show VM state
	@bash $(GCP) status

gcp-ssh: ## Interactive SSH into the VM
	@bash $(GCP) ssh

gcp-run: ## Full cycle: up -> train -> down (VM is stopped even if up/training fails)
	@bash $(GCP) up; rc=$$?; \
	if [ $$rc -eq 0 ]; then bash $(GCP) train; rc=$$?; fi; \
	bash $(GCP) down; exit $$rc


gcp-run-cpu: ## Full cycle on a Spot CPU VM (no GPU quota needed)
	@GCE_MODE=cpu bash $(GCP) up; rc=$$?; \
	if [ $$rc -eq 0 ]; then GCE_MODE=cpu bash $(GCP) train; rc=$$?; fi; \
	GCE_MODE=cpu bash $(GCP) down; bash scripts/session_sync.sh push || true; exit $$rc

# ----------------------------------------------------------------- session backup
sessions-push: ## Back up Kiro/Gemini/ADK sessions + logs + submissions to $(GCS_BUCKET_URI)/sessions
	@bash scripts/session_sync.sh push

sessions-pull: ## Restore sessions/logs from the bucket (never overwrites newer local files)
	@bash scripts/session_sync.sh pull

sessions-status: ## List backed-up session files
	@bash scripts/session_sync.sh status
