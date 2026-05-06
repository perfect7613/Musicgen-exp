#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev --extra audio
else
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  pip install -e ".[dev,audio]"
fi

echo "Base experiment environment installed."
echo "Next: install AudioCraft/musicdiscovery dependencies according to their current upstream docs."
