# Security policy

## Supported versions

| Version | Supported |
|---|---|
| `0.1.x` (latest release) | Yes |
| Older releases | No |

Security fixes are released as a new patch version; the repository is
append-only and published tags are never moved.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
on this repository ("Security" tab, "Report a vulnerability") and include:

- the affected version or commit,
- a minimal reproduction,
- the impact you observed.

You can expect an acknowledgement within a few days. Please give a reasonable
window for a fix before public disclosure.

## Scope

The project runs locally and collects no telemetry. It requires no API keys or
accounts; `rdv models pull` and `rdv data download` are the only commands that
make network requests, and `RDV_OFFLINE=1` disables them. Areas worth reporting:

- the local HTTP server: `Host`/`Origin` validation, upload size limits, MIME
  and decode validation, temporary-file cleanup, path handling;
- archive parsing: the ranged nested-ZIP reader, CRC32/sha256 verification and
  resumable downloads;
- model loading: manifest validation, digest checks and ONNX parsing;
- dependency or build supply chain issues.

The server binds `127.0.0.1` only, rejects foreign `Host` headers (421),
rejects cross-origin state-changing requests (403), and never derives a
filesystem path from a user-supplied filename. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/RELEASING.md](docs/RELEASING.md) for the intended behavior.

## Not in scope

- Accuracy, false positives/negatives or the model's fitness for a purpose.
  The tool is not safety-critical and must not be the only basis for road
  maintenance, traffic-safety, insurance or legal decisions.
- The unsigned macOS/Windows desktop warnings (Gatekeeper/SmartScreen). The
  builds are intentionally unsigned and documented in
  [docs/RELEASING.md](docs/RELEASING.md).
- Vulnerabilities in optional local tooling outside the committed lockfile.

## Supply chain

CI runs gitleaks over the full history, `pip-audit` over the locked runtime
export and a license guard that rejects AGPL/GPL runtime dependencies. Build
artifacts are attested with GitHub provenance on tagged releases. No secrets are
stored in the repository; the workflows only use `GITHUB_TOKEN`.
