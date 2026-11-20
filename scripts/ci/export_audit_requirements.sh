#!/usr/bin/env bash
# Export the locked, torch-free runtime requirements for pip-audit (S17).
#
# Usage: scripts/ci/export_audit_requirements.sh [OUTPUT]
# Writes to stdout unless an output path is given.
set -euo pipefail

output="${1:-}"
args=(
  export
  --locked
  --no-dev
  --no-default-groups
  --format requirements-txt
  --no-emit-project
  --no-annotate
  --no-header
)

if [ -n "${output}" ]; then
  uv "${args[@]}" -o "${output}"
else
  uv "${args[@]}"
fi
