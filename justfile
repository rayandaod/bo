default:
    just --list

# Check linting
lint:
    uv run ruff format
    uv run ruff check --fix

# Run tests
test pytest_args='-svvv':
    uv run pytest {{pytest_args}}
