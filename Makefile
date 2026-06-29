# Convenience targets for neuronauts development.
# `make setup` once, then `make test` / `make test-fast`.

.PHONY: setup test test-fast lint clean

# Full dev install (editable + dev/topology/cave extras).
setup:
	pip install -r requirements-dev.txt

# Whole suite. Tolerates collection errors so a missing `cave` extra only skips
# the caveclient tests instead of aborting the run.
test:
	pytest -q --continue-on-collection-errors

# Skip the slow v1 simulation tests (the `legacy` marker).
test-fast:
	pytest -q -m 'not legacy' --continue-on-collection-errors

# Remove caches and build artifacts (never touches data/ or models/).
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info \
	  neuronauts.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
