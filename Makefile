# hcm-twin
#
# `make all` is the definition of done: from a clean checkout it runs the test suite
# including every validation gate, generates the virtual population from a logged seed,
# and produces every table, figure, the interactive explorer and the compiled writeup.
#
# `make docker-all` does the same inside the container, which is the reproducible form.

PY      := .venv/bin/python
PIP     := uv pip install --python $(PY)
RESULTS := results
PAPER   := paper

# Study size. Override for a quick pass, e.g.
#   make all N_BASE=48 CASES=8 DESIGN=200 MCMC=800
N_BASE  ?= 385
CASES   ?= 50
DESIGN  ?= 400
MCMC    ?= 3000

.DEFAULT_GOAL := help
.PHONY: help setup test test-fast lint typecheck validate population figures \
        sensitivity identifiability tiebreaker explorer paper all notebooks \
        clean clean-results docker docker-all

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------

setup:  ## Create the virtual environment and install everything
	uv venv --python 3.12 .venv
	$(PIP) -e '.[dev]'

# ---------------------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------------------

test:  ## Full test suite, including every Section 7 validation gate
	$(PY) -m pytest -q

test-fast:  ## Everything except the multi-minute cohort tests
	$(PY) -m pytest -q -k "not population"

lint:  ## Style and static checks
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

typecheck:  ## Type checks
	.venv/bin/mypy

# ---------------------------------------------------------------------------------------
# Stages. Each writes into results/ and later stages read from there.
# ---------------------------------------------------------------------------------------

validate:  ## D2: the validation table
	$(PY) -m hcmtwin.cli validate --results $(RESULTS)

population:  ## The virtual cohort
	$(PY) -m hcmtwin.cli population --results $(RESULTS) --n-base $(N_BASE)

figures:  ## Reference figures: loops, dose response, the premise, the noise budget
	$(PY) -m hcmtwin.cli figures --results $(RESULTS) --n-base $(N_BASE)

sensitivity:  ## D3: the sensitivity matrix
	$(PY) -m hcmtwin.cli sensitivity --results $(RESULTS) --n-base $(N_BASE)

identifiability:  ## D4: the confounding map
	$(PY) -m hcmtwin.cli identifiability --results $(RESULTS) --n-base $(N_BASE) \
	  --cases $(CASES) --design $(DESIGN) --mcmc-steps $(MCMC)

tiebreaker:  ## D5: the tie-breaker table
	$(PY) -m hcmtwin.cli tiebreaker --results $(RESULTS) --n-base $(N_BASE) \
	  --cases $(CASES) --design $(DESIGN) --mcmc-steps $(MCMC)

explorer:  ## D6: the self-contained interactive page
	$(PY) -m hcmtwin.cli explorer --results $(RESULTS) --n-base $(N_BASE)

paper:  ## D7: regenerate the paper's numbers from results/ and compile
	$(PY) -m hcmtwin.cli paper --results $(RESULTS)

notebooks:  ## Execute the notebooks in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

all:  ## Everything, from a clean checkout
	$(PY) -m pytest -q
	$(PY) -m hcmtwin.cli all --results $(RESULTS) --n-base $(N_BASE) \
	  --cases $(CASES) --design $(DESIGN) --mcmc-steps $(MCMC)
	@echo
	@echo "Deliverables:"
	@echo "  D2 validation table   $(RESULTS)/validation_table.md"
	@echo "  D3 sensitivity        $(RESULTS)/fig_sensitivity_matrix.png (+ .csv)"
	@echo "  D4 confounding map    $(RESULTS)/fig_confounding_map.png (+ .csv)"
	@echo "  D5 tie-breaker table  $(RESULTS)/tiebreaker_table.csv"
	@echo "  D6 explorer           $(RESULTS)/explorer.html"
	@echo "  D7 writeup            $(PAPER)/main.pdf"

# ---------------------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------------------

clean:  ## Remove build and cache artefacts (keeps results/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ \
	       $(PAPER)/*.aux $(PAPER)/*.log $(PAPER)/*.out $(PAPER)/*.toc

clean-results:  ## Remove every generated result. Destructive; asks nothing.
	rm -rf $(RESULTS) $(PAPER)/generated.tex $(PAPER)/table_*.tex $(PAPER)/main.pdf

# ---------------------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------------------

docker:  ## Build the reproducible image
	docker build -t hcm-twin .

docker-all: docker  ## Run the whole study inside the container
	docker run --rm \
	  -v "$(CURDIR)/results:/app/results" \
	  -v "$(CURDIR)/paper:/app/paper" \
	  hcm-twin
