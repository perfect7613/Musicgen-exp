#!/usr/bin/env bash
set -euo pipefail

echo "RunPod account:"
runpodctl user

echo
echo "RunPod pods:"
runpodctl pod list --all

echo
echo "Available GPUs:"
runpodctl gpu list

echo
echo "Preflight complete. Do not create a pod unless current spend is 0 and the planned run fits the budget."
