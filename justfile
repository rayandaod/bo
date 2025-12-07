default:
    just --list

# Check linting
lint:
    uv run ruff format
    uv run ruff check --fix

# Run tests
test test_folder='tests' pytest_args='-svvv':
    uv run pytest {{test_folder}} {{pytest_args}}
