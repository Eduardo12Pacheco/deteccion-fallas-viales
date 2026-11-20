# 0008 - Model distribution through a pinned manifest and a resumable store

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D13, D22, S20)
- **Task:** T6 (store and CLI), T10 (first published artifact)

## Context

The wheel must stay small and the repository must never commit model weights.
At the same time `rdv detect` has to work offline once a model is present, must
not download anything implicitly (spec S20), and must be able to prove that
the file it executes is the exact artifact the metrics in the README were
produced with. Weights are produced later, on the owner's GPU, and published
as a GitHub Release asset (`models-v1`).

## Decision

- `src/rdv/assets/models.json` (schema `rdv.models/v1`) is the single source of
  truth for downloadable artifacts. Each entry pins `id`, `version`, `url`,
  `sha256`, `size`, `input_size`, `recommended_score_threshold`, `classes`,
  `license` and an optional `metrics_ref`. The list is empty until T10
  publishes the first fine-tune.
- `rdv models list|pull|path|verify` is the only surface that downloads or
  validates weights:
  - `pull` streams to `<name>.part`, resumes with an HTTP `Range` request
    when a partial file exists (restarting from zero if the server ignores the
    range), verifies the exact byte size and sha256, and only then renames the
    file atomically. A sha256 mismatch removes the partial file and exits 4.
  - `RDV_OFFLINE=1` makes `pull` fail before any HTTP request.
  - `resolve` accepts either a manifest id (looked up in `RDV_MODEL_DIR`) or an
    explicit `.onnx` path, so local models and published models share one
    code path.
  - `verify` compares a file with its manifest entry (or finds the entry whose
    digest matches) and returns that entry.
- `rdv detect` resolves the model through the same store. A missing model is
  a `ModelError` (exit 4) whose message points at `rdv models pull`; there is
  no silent download.
- The recommended threshold comes from the manifest; `RDV_SCORE_THRESHOLD`
  overrides it and `--conf` overrides both. Without a manifest entry (local
  file) the documented fallback is `0.25`.

## Consequences

- The repository, wheel and desktop bundle contain no weights; the desktop
  bundle embeds the artifact at build time (T13) after verifying this same
  sha256.
- Every detection document records the model id and sha256, so a report can be
  traced to the exact artifact.
- Downloads are idempotent: a verified cache entry short-circuits the network,
  and a network failure keeps the `.part` for the next attempt.
- No key is required for a public release. Because the portfolio repository is
  private, `pull` also honors an optional `GITHUB_TOKEN`/`GH_TOKEN` (for
  example `gh auth token`) and sends the bearer header only to GitHub hosts;
  anonymous requests are unchanged.
- The manifest is small enough to review by hand, which keeps the trust anchor
  visible in the repository instead of hidden in release metadata.
