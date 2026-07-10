.PHONY: install lint format typecheck test smoke check

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	black .
	isort .

typecheck:
	mypy --strict src/oncosentinel

test:
	pytest -q

smoke:
	fm-mapb orchestrate --experiment _smoke

check: lint typecheck test
