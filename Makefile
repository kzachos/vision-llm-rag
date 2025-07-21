SHELL :=/bin/bash

.PHONY: clean check setup
.DEFAULT_GOAL=help
VENV_DIR = .venv
PYTHON_VERSION=python3.11

check: # Run Ruff linter on all code
	@ruff check .
	@echo "\u2705 Lint check complete!"

fix: # Auto-fix linting issues in app.py
	@ruff check app.py --fix

clean: # Remove temporary and build files
	@rm -rf __pycache__ .pytest_cache
	@find . -name '*.pyc' -exec rm -r {} +
	@find . -name '__pycache__' -exec rm -r {} +
	@rm -rf build dist
	@find . -name '*.egg-info' -type d -exec rm -r {} +

run: # Run the RAG QnA platform
	@streamlit run app.py

setup: # Set up virtual environment and install dependencies
	@echo "Creating virtual env at: $(VENV_DIR)"
	@$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	@source $(VENV_DIR)/bin/activate && pip install -r requirements/requirements-dev.txt && pip install -r requirements/requirements.txt
	@echo -e "\n\u2705 Setup complete.\n\ud83c\udf89 To get started:\n\n \u27a1\ufe0f source $(VENV_DIR)/bin/activate\n \u27a1\ufe0f make run\n"

help: # Show this help message
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
