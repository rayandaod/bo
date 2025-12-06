default:
    just --list

# Check linting
lint:
    uv run ruff check

# Run tests
test pytest_args='':
    uv run pytest {{pytest_args}}
