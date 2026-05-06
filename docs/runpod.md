# RunPod Workflow

This project uses `runpodctl` for remote GPU work. Pod creation is intentionally explicit so credits are not spent accidentally.

## 1. Authenticate

```bash
runpodctl doctor
```

If GitHub or shell authentication is stale, fix that locally before creating pods.

## 2. Inspect Available GPUs

```bash
runpodctl gpu list
runpodctl template search pytorch --type official --limit 10
```

Recommended first choices:

- RTX 4090 / 5090 for MusicGen-Small debugging.
- A40 / A100 for MusicGen-Large and larger activation sweeps.

## 3. Create A Pod

Example:

```bash
runpodctl pod create \
  --name musicgen-exp \
  --template-id runpod-torch-v21 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --container-disk-in-gb 80 \
  --volume-in-gb 100 \
  --ports "8888/http,22/tcp"
```

Use the exact `--template-id` and `--gpu-id` returned by your RunPod account. Prices and availability change frequently.

## 4. Transfer This Repo

From the local repo root:

```bash
tar -czf musicgen-exp.tar.gz .
runpodctl send musicgen-exp.tar.gz
```

On the pod, receive the file with the code phrase printed by `runpodctl send`, then unpack it into `/workspace/musicgen-exp`.

## 5. Remote Setup

On the pod:

```bash
cd /workspace/musicgen-exp
bash scripts/setup_remote.sh
```

## 6. Shutdown Discipline

Always stop or terminate pods when finished. GPU notebooks are excellent at quietly eating credits.
