## Summary

<!-- One paragraph: what changes and why. Link the issue when one exists. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor or architecture (no behavior change)
- [ ] Documentation, tests or tooling
- [ ] Packaging or release

## How it was tested

<!-- Exact commands and observed results. -->

```bash
uv run pytest -q
make ci-local
```

## Checklist

- [ ] I read `CONTRIBUTING.md` and the architecture notes in `docs/`.
- [ ] Tests were written first and cover the new behavior or the regression.
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy --strict src`
      and `uv run lint-imports` pass.
- [ ] The changelog or the docs were updated when behavior changed.
- [ ] No secrets, private data, AI attribution or absolute personal paths are included.
- [ ] Large artifacts (datasets, weights, bundles) are not part of this diff.
