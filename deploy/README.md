# deploy/

Cloud deployment artifacts. The local Mac dev path uses `start.sh` at the repo root; this folder is only relevant for putting VisualRef on a remote VM.

## Layout

- `Dockerfile` — server image (CUDA base; runs the FastAPI backend only).
- `setup-cloud.sh` — invoked on a fresh VM. Clones the repo, runs `scripts/cloud_bootstrap.sh`, starts the server.
- `env.cloud` — environment template (model paths, ports, GPU flags) sourced by the bootstrap script.
- `DEPLOY.md` — step-by-step deploy walkthrough.
