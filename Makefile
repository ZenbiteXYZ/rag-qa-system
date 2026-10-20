COMPOSE := docker compose -f deploy/docker-compose.yml --env-file deploy/.env
PROFILES ?=
PROFILE_FLAGS := $(foreach p,$(PROFILES),--profile $(p))

.PHONY: up down logs ps test test-integration lint format typecheck check

up:  ## Start services. Extra profiles: make up PROFILES="ml obs"
	$(COMPOSE) $(PROFILE_FLAGS) up -d --build

down:  ## Stop services; volumes are kept
	$(COMPOSE) --profile ml --profile obs down

logs:  ## Follow logs: make logs S=api
	$(COMPOSE) $(PROFILE_FLAGS) logs -f $(S)

ps:
	$(COMPOSE) --profile ml --profile obs ps

test:
	uv run pytest

test-integration:  ## Needs the broker up and RAGQA_TEST_RABBITMQ_URL set
	uv run pytest -m integration

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test
