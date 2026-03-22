.PHONY: setup run-all run-01 run-02 run-03 run-04 run-05 lint format clean

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

setup: $(VENV)/bin/activate  ## Create venv and install dependencies
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev]"

run-all: run-01 run-02 run-03 run-04 run-05  ## Run all demos

run-01: ## I-JEPA representations (clustering & similarity)
	$(PY) demos/01_ijepa_representations.py

run-02: ## I-JEPA masking explainer (no model needed)
	$(PY) demos/02_ijepa_masking_explained.py

run-03: ## V-JEPA 2 video classification
	$(PY) demos/03_vjepa_video_classify.py

run-04: ## V-JEPA 2 action anticipation
	$(PY) demos/04_vjepa_action_anticipation.py

run-05: ## V-JEPA 2 temporal cluster analysis
	$(PY) demos/05_vjepa_cluster_analysis.py

lint: ## Check code style
	$(VENV)/bin/ruff check demos/

format: ## Auto-format code
	$(VENV)/bin/ruff format demos/
	$(VENV)/bin/ruff check --fix demos/

clean: ## Remove generated outputs and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -f outputs/*.png outputs/*.jpg outputs/*.mp4

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help