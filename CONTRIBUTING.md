# Contributing

Thanks for taking a look. This is a portfolio project, but issues and pull
requests are welcome as long as they keep the quality gates green.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/) and Git. Python 3.12 is
provisioned by uv.

```bash
uv sync --extra train --extra cpu    # full environment (training stack, CPU torch)
make ci-local                        # every CI job command, including clean-install smoke
```

The base `uv sync` (no extras) is enough for inference, the CLI, the web app and
the torch-free test matrix.

## Checks

| Command | What it enforces |
|---|---|
| `uv run ruff check .` / `uv run ruff format --check .` | Lint and formatting |
| `uv run mypy --strict src` | Strict typing on every source file |
| `uv run lint-imports` | Hexagonal layer, purity and adapter-independence contracts |
| `uv run pytest -q` | Unit, integration and CLI/API end-to-end tests |
| `uv run pytest -q -m ui` | Playwright browser flows (localhost only) |
| `uv run pytest -q --cov=src/rdv --cov-fail-under=85` | Coverage gate |
| `uv run pytest tests/unit/test_docs_links.py -q` | Every relative link and `#anchor` resolves |
| `uv lock --check` | The lockfile matches `pyproject.toml` |

Tests are deterministic and offline by default (`pytest-socket`); use
`httpx.MockTransport` or committed fixtures for network adapters. Markers:
`train` (needs torch), `ui` (Playwright), `parity` (`pycocotools`) and
`realmodel` (never in CI).

## Conventions

- Conventional Commits in English, one coherent change per commit, no AI
  attribution trailers.
- Test first: add the failing test, then the implementation, then refactor.
- Keep the dependency direction `entrypoints > composition > adapters >
  application > domain`; `tests/unit/test_architecture.py` enforces it.
- Pure logic belongs in `rdv.domain` (stdlib + numpy only); third-party I/O
  belongs in an adapter and is wired in `composition.py`.
- Do not commit datasets, model weights, run directories, absolute paths or
  secrets. Fixtures and samples follow the CC BY-SA 4.0 stance in
  [docs/DATA.md](docs/DATA.md).
- Update the relevant doc when behavior changes; record a significant
  decision as a new ADR in [docs/adr/](docs/adr/README.md).
- Code, comments and UI strings are in English; `README.es.md` and Spanish
  class labels/UI translations are the exceptions.

## Pull requests

Keep the diff focused, describe the behavior change and list the commands you
ran. A pull request is ready when `make ci-local` is green locally and the
README/docs still resolve. Security reports go through the process in
[SECURITY.md](SECURITY.md), not a public issue.
