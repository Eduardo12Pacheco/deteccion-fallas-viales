#!/usr/bin/env bash
# Publish the pinned ONNX artifact to the models-v1 GitHub Release (D13/D22).
#
# The manifest is the trust anchor: this script refuses to upload a file whose
# sha256/size differ from src/rdv/assets/models.json, then creates the release
# once or uploads over the existing asset so a re-run is idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${1:-rdv-rtdetrv2-r18-rdd2022}"
ARTIFACT="${2:-artifacts/${MODEL_ID}-v1.onnx}"
RELEASE_TAG="models-v1"
MANIFEST="src/rdv/assets/models.json"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh is required to publish the model release" >&2
  exit 1
fi
if [[ ! -f "$ARTIFACT" ]]; then
  echo "error: artifact not found: $ARTIFACT" >&2
  exit 1
fi

python - "$MANIFEST" "$MODEL_ID" "$ARTIFACT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, model_id, artifact_path = sys.argv[1:4]
payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
entries = {entry["id"]: entry for entry in payload.get("models", [])}
entry = entries.get(model_id)
if entry is None:
    raise SystemExit(f"error: {model_id} is not in {manifest_path}")
artifact = Path(artifact_path)
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
size = artifact.stat().st_size
if digest != entry["sha256"] or size != int(entry["size"]):
    raise SystemExit(
        "error: artifact does not match the manifest "
        f"(sha256 {digest} size {size}, expected {entry['sha256']} {entry['size']})"
    )
print(f"verified {artifact}: sha256 {digest} ({size} bytes)")
PY

if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
  gh release upload "$RELEASE_TAG" "$ARTIFACT" --clobber
  echo "uploaded $ARTIFACT to existing release $RELEASE_TAG"
else
  NOTES="Pinned RT-DETRv2 R18 fine-tuned on RDD2022 (Japan/India/Czech, 4 classes). Single-file ONNX, opset 18, input pixel_values [N,3,640,640] in [0,1]. sha256 and size are pinned in src/rdv/assets/models.json; the file is Apache-2.0 code with CC BY-SA 4.0 dataset derivatives (see docs/DATA.md)."
  gh release create "$RELEASE_TAG" "$ARTIFACT" \
    --title "RDD2022 RT-DETRv2 R18 weights (v1)" \
    --notes "$NOTES"
  echo "created release $RELEASE_TAG with $ARTIFACT"
fi
