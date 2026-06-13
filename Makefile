.PHONY: lint format format-check typecheck test test-prkit build check

lint:
	python -m ruff check src/prkit tests/prkit

format:
	python -m ruff format src/prkit tests/prkit
	python -m black src/prkit tests/prkit

format-check:
	python -m black --check src/prkit tests/prkit

typecheck:
	python -m mypy src/prkit

test:
	python -m pytest

test-prkit:
	python -m pytest tests/prkit

build:
	python -m build

check: lint format-check test-prkit
