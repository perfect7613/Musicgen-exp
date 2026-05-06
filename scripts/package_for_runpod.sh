#!/usr/bin/env bash
set -euo pipefail

archive_name="${1:-musicgen-exp.tar.gz}"

tar \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="outputs/audio/*" \
  --exclude="outputs/figures/*" \
  -czf "$archive_name" .

echo "Created $archive_name"
echo "Send with: runpodctl send $archive_name"
