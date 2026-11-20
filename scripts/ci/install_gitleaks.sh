#!/usr/bin/env bash
# Install the pinned gitleaks binary with checksum verification (D6, S17).
#
# Usage: scripts/ci/install_gitleaks.sh [DEST_DIR]
# Environment: GITLEAKS_VERSION (default 8.30.1)
set -euo pipefail

version="${GITLEAKS_VERSION:-8.30.1}"
dest="${1:-${HOME}/.local/bin}"
base="https://github.com/gitleaks/gitleaks/releases/download/v${version}"

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${os}" in
  linux) platform="linux" ;;
  darwin) platform="darwin" ;;
  mingw* | msys* | cygwin*) platform="windows" ;;
  *)
    echo "unsupported operating system: ${os}" >&2
    exit 1
    ;;
esac

machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
case "${machine}" in
  x86_64 | amd64) arch="x64" ;;
  aarch64 | arm64) arch="arm64" ;;
  armv6l) arch="armv6" ;;
  armv7l) arch="armv7" ;;
  *)
    echo "unsupported architecture: ${machine}" >&2
    exit 1
    ;;
esac

if [ "${platform}" = "windows" ]; then
  asset="gitleaks_${version}_${platform}_${arch}.zip"
else
  asset="gitleaks_${version}_${platform}_${arch}.tar.gz"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
curl -fsSL --retry 3 --retry-delay 2 -o "${tmp}/${asset}" "${base}/${asset}"
curl -fsSL --retry 3 --retry-delay 2 -o "${tmp}/checksums.txt" "${base}/gitleaks_${version}_checksums.txt"

(
  cd "${tmp}"
  grep -E "  ${asset}\$" checksums.txt > expected.txt
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c expected.txt
  else
    shasum -a 256 -c expected.txt
  fi
)

mkdir -p "${dest}"
if [ "${platform}" = "windows" ]; then
  unzip -q -o "${tmp}/${asset}" -d "${tmp}/unpacked"
  install -m 0755 "${tmp}/unpacked/gitleaks.exe" "${dest}/gitleaks.exe"
  "${dest}/gitleaks.exe" version
  echo "${dest}/gitleaks.exe"
else
  tar -xzf "${tmp}/${asset}" -C "${tmp}"
  install -m 0755 "${tmp}/gitleaks" "${dest}/gitleaks"
  "${dest}/gitleaks" version
  echo "${dest}/gitleaks"
fi
