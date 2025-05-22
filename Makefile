#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = thesis
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: black
black:
	black --config pyproject.toml sltns

## Format source code with ruff
.PHONY: ruff
ruff:
	ruff format .

## Lint source code with ruff
.PHONY: lint
lint:
	ruff check .

## Run pre-commit hooks
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

## Create docs
.PHONY: docs
docs:
	mkdocs serve

## Set up python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) periomod/data/_preprocessing.py ${ARGS}

.PHONY: benchmark
benchmark:
	$(PYTHON_INTERPRETER) periomod/benchmarking/_benchmark.py ${ARGS}
