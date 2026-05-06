#!/usr/bin/env bash
set -euo pipefail

# Copy this file to create_runpod_pod.local.sh and edit the template/gpu IDs.
# Do not commit local credentials or account-specific pod IDs.

runpodctl pod create \
  --name musicgen-exp \
  --template-id runpod-torch-v21 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --container-disk-in-gb 80 \
  --volume-in-gb 100 \
  --ports "8888/http,22/tcp"
