.PHONY: install dev test test-unit test-integration lint typecheck fmt clean services-up services-down services-check

install:
	uv sync

dev:
	uv sync --extra dev

test:
	uv run pytest tests/unit/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-all:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy libs/ orchestrators/ apps/cli/ --strict

fmt:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info

# --- Docker services ---

services-up:
	docker compose up -d
	@echo "Waiting for services to become healthy..."
	@sleep 5
	@./scripts/check_services.sh

services-down:
	docker compose down

services-check:
	@./scripts/check_services.sh
