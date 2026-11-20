# Releasing road-damage-vision

This project ships two independent release channels (decision D22):

| Channel | Tag | Assets | Purpose |
|---------|-----|--------|---------|
| Model weights | `models-v1` | `rdv-rtdetrv2-r18-rdd2022-v1.onnx` | Pinned ONNX file downloaded by `rdv models pull` and embedded in desktop bundles |
| Application | `v0.1.0` | wheel, sdist, three desktop bundles, `SHA256SUMS`, provenance attestations | CLI (L1) and portable desktop app (L3) |

The weights release is created **once** by the repository owner and is never
rebuilt by the tag workflow. The tag workflow downloads it with
`gh release download`, verifies its sha256 against
`src/rdv/assets/models.json`, embeds it in every bundle and re-uses the same
digest in the post-release smoke. If the two digests disagree, the workflow
stops before publishing anything.

## Before the first release

1. Confirm that the remote is reachable and CI is green on `main`.
2. Publish the weights release (owner action, exactly once):

   ```bash
   bash scripts/publish_model.sh
   ```

   The script refuses an artifact whose sha256 or size differs from the
   manifest. Without this release the bundle jobs and `rdv models pull` fail.
3. Confirm the manifest pins a complete entry:

   ```bash
   uv run python scripts/ci/verify_release_inputs.py v0.1.0
   ```

## Release order

1. **Bump the version** in `pyproject.toml` (and refresh `uv.lock` only if a
   dependency changed).
2. **Regenerate the changelog** from Conventional Commits:

   ```bash
   uvx git-cliff@2.14.2 --tag v0.1.0 -o CHANGELOG.md
   ```

   Inspect the section; commit it with the version bump.
3. **Run the local preflight** (the same checks the workflow runs):

   ```bash
   uv lock --check
   uv run ruff check . && uv run ruff format --check .
   uv run mypy --strict src
   uv run lint-imports
   uv run pytest -q
   uv run python scripts/ci/verify_release_inputs.py v0.1.0
   uvx git-cliff@2.14.2 --tag v0.1.0 --unreleased > /dev/null
   ```

4. **Tag and push** the annotated tag. Pushing the tag is what starts
   `.github/workflows/release.yml`; the tag must equal `v` + the
   `pyproject.toml` version or the `verify` job fails before building.

   ```bash
   git tag -a v0.1.0 -m "road-damage-vision 0.1.0"
   git push origin v0.1.0
   ```

5. **Watch the workflow**:

   ```bash
   gh run list --workflow release.yml --limit 3
   gh run watch <run-id> --exit-status
   ```

## What the workflow does

| Job | Result |
|-----|--------|
| `verify` | Tag equals the project version; at least one model pins a 64-character sha256, size, URL, threshold and license |
| `dist` | `uv build` produces the wheel and sdist; `twine check` validates both |
| `bundles` | Matrix `ubuntu-24.04` → `linux-x64`, `windows-latest` → `windows-x64`, `macos-latest` → `macos-arm64`; downloads and verifies `models-v1`, builds the PyInstaller onedir bundle, runs `scripts/smoke_bundle.py`, stages `RoadDamageVision-<version>-<platform>.{tar.gz,zip}` |
| `release` | Generates notes with `git-cliff --latest --strip header`, computes `SHA256SUMS`, attests provenance for every asset, publishes the release |
| `post-release-smoke` | Installs the published wheel with `uv tool install`, pulls the real model (`rdv models pull` verifies sha256), runs `scripts/smoke_release.py` against `reports/v1/sample_expectations.json` |

The post-release smoke requires every expected class to be present with a
score at or above the pinned threshold. The current `sample_expectations.json`
records `margin_ok: false` for all six samples because the v1 head is
under-confident; the smoke therefore asserts class presence and threshold, not
the 0.15 margin described in `docs/MODEL_CARD.md`.

## Verify a download

Users can verify the exact bytes they downloaded:

```bash
sha256sum -c SHA256SUMS
gh attestation verify road_damage_vision-0.1.0-py3-none-any.whl \
  --repo Eduardo12Pacheco/road-damage-vision
```

On Windows, use `Get-FileHash -Algorithm SHA256 <file>` and compare with the
matching line in `SHA256SUMS`.

The desktop archives are **unsigned**. macOS blocks them with Gatekeeper until
the user allows the app in *System Settings → Privacy & Security*, and Windows
shows SmartScreen until the user chooses *More info → Run anyway*. The Linux
bundle requires glibc 2.39 or newer because it is built on Ubuntu 24.04.

## Rollback

- Never move or reuse a published tag: releases are append-only.
- If an asset is wrong, publish a patch version (`v0.1.1`). As an emergency
  measure an owner may delete the GitHub Release and its tag, but the tag must
  not be recreated; the next attempt gets a new version number.
- If the weights asset is corrupt, re-upload it only if the manifest sha256 is
  unchanged (`scripts/publish_model.sh` is idempotent and checks the digest).
  A new model needs a new `version` in the manifest and a new `models-v2`
  release, plus a bundle rebuild.
- If the post-release smoke fails, fix forward with a patch release; do not
  edit `reports/v1/sample_expectations.json` to make the smoke pass (decision
  D17 keeps published numbers unchanged).

## Private-repository note

The repository is private, so release assets need authentication. `gh` uses
`GH_TOKEN`/`GITHUB_TOKEN` automatically. `rdv models pull` also accepts either
variable and resolves the release asset through the GitHub API; without a token
the download fails with a clear message instead of an anonymous 404. The
post-release smoke therefore exports `GITHUB_TOKEN` for the pull step.

## Troubleshooting

- `tag vX does not match pyproject version Y`: bump `pyproject.toml`, commit,
  delete the local tag with `git tag -d vX`, and tag the corrected commit.
- `model asset ... does not match the manifest`: `models-v1` was rebuilt or the
  manifest digest changed. Re-run `scripts/publish_model.sh` from the artifact
  that matches the committed manifest.
- Bundle smoke timeout: inspect the failing job log; the script prints the last
  30 lines of the launcher log and enforces a 90 s health timeout.
- Release job cannot attest: check that the workflow kept
  `id-token: write` and `attestations: write`; both are required by
  `actions/attest-build-provenance`.
