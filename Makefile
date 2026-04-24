.PHONY: format

format:
	uv run ruff format .
	uv run ruff check --fix .
