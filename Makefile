.PHONY: lint format format-check typecheck test test-prkit build check ci

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

check: lint format-check typecheck test-prkit

# Faithful local mirror of .github/workflows/ci.yml: runs every gate the same
# way CI does, in a CI-like environment with NO .env and NO OPENAI_API_KEY, so
# tests that secretly rely on a local key fail here instead of in CI. The .env
# is moved aside and restored afterwards even if a gate fails. Run before pushing.
ci:
	@bash -c 'set -u; \
	  if [ -f .env ]; then mv .env .env.cibak; fi; \
	  trap "[ -f .env.cibak ] && mv .env.cibak .env" EXIT; \
	  unset OPENAI_API_KEY; \
	  python -m ruff check src/prkit tests/prkit && \
	  python -m black --check src/prkit tests/prkit && \
	  python -m mypy src/prkit && \
	  python -m pytest tests/prkit'
