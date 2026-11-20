# Contributing

Thanks for taking a look. This is a portfolio project, but issues and pull
requests are welcome as long as they keep the quality gates green.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/) and Git. Python 3.12 is
provisioned by uv.

```bash
uv sync
uv run pytest -q
```

Run the checks that continuous integration runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run lint-imports
uv run pytest -q
```

The full suite (training and browser tests) needs optional extras and browser
binaries; see `Makefile` for the available targets.

## Conventions

- Conventional Commits in English, one coherent change per commit.
- Test first: add the failing test, then the implementation.
- Keep the hexagonal dependency direction; `tests/unit/test_architecture.py`
  enforces it.
- Do not commit datasets, model weights, run directories, or secrets.
- Code, comments, and UI strings are in English.
