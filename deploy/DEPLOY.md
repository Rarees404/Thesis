# Cloud GPU Deployment — VisualRef

## What You Need

| Resource | Size | Purpose |
|----------|------|---------|
| Visual Genome images + JSON | ~15 GB | Image corpus for retrieval |
| FAISS index (VG, SigLIP) | ~hundreds of MB | Pre-built embeddings (build on instance or upload) |
| SAM3 model | ~2.6 GB | Auto-downloaded from HuggingFace |
| SigLIP model | ~1.1 GB | Auto-downloaded from HuggingFace |
| Ollama + llama3.2-vision | ~4.7 GB | Optional VLM captioning |
| **Total disk** | **~20+ GB** | VG + models + index |

**GPU requirement**: Any NVIDIA GPU with ≥16 GB VRAM (RTX 3090, RTX 4090, A100, etc.)

---

## Cheapest Option: Vast.ai (~$0.15–0.25/hr)

### Step 1: Create account
Go to [vast.ai](https://vast.ai/) and add $5–10 credit (enough for 20–60 hours).

### Step 2: Rent a GPU instance
1. Click **Search** → filter by:
   - GPU: **RTX 3090** (24 GB, cheapest that works)
   - Disk: **50 GB** (enough for data + models)
   - Image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
2. Pick the cheapest one (usually $0.12–0.20/hr)
3. Click **Rent**

### Step 3: Connect via SSH
Vast.ai gives you an SSH command like:
```bash
ssh -p 12345 root@ssh5.vast.ai
```

### Step 4: Upload your project + data
From your Mac terminal:
```bash
# Upload the project code (small, fast)
rsync -avz --exclude='node_modules' --exclude='venv' --exclude='__pycache__' \
  -e "ssh -p 12345" \
  /Users/blackbox/Thesis_Proto/visualref/ \
  root@ssh5.vast.ai:/workspace/visualref/

# Upload Visual Genome data (large — or run download_visual_genome.sh on the instance)
rsync -avz -e "ssh -p 12345" \
  /path/to/visualref/data/visual_genome/ \
  root@ssh5.vast.ai:/workspace/visualref/data/visual_genome/

# Optional: upload a pre-built FAISS tree, or build on the GPU with scripts/build_index.sh
rsync -avz -e "ssh -p 12345" \
  /path/to/visualref/faiss/visual_genome/ \
  root@ssh5.vast.ai:/workspace/visualref/faiss/visual_genome/
```

### Step 5: Set up the server on the GPU
SSH into the instance and run:
```bash
cd /workspace/visualref
bash deploy/setup-cloud.sh
```

If SAM3 needs HuggingFace auth:
```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 6: Start the server
```bash
cd /workspace/visualref/server
source venv/bin/activate
python -m uvicorn src.retrieval_server_visual:app --host 0.0.0.0 --port 8001
```

You should see:
```
[startup] SAM backend: sam3 (requested: sam3)
[startup] Ollama vision: ...
```

### Step 7: Connect your local frontend
The Vast.ai instance has a public IP. On your Mac:

1. Edit `client-next/.env.local`:
   ```
   NEXT_PUBLIC_SERVER_URL=http://<vast-ip>:8001
   ```
   Replace `<vast-ip>` with the instance IP from Vast.ai dashboard.

2. Start the frontend locally:
   ```bash
   cd client-next && npm run dev
   ```

3. Open http://localhost:3000 — it now talks to the cloud GPU.

---

## Alternative: RunPod (~$0.22–0.44/hr)

Same steps, but use [runpod.io](https://www.runpod.io/):
- Select **Community Cloud** for cheapest rates
- Choose **RTX 3090** or **RTX 4090**
- Use the web terminal or SSH

---

## To run locally

Edit `server/.env`:
```
SAM_BACKEND=sam3
```

Run locally with `./start.sh` as before.

---

## Cost estimate

| Usage | Vast.ai (RTX 3090) | RunPod (RTX 3090) |
|-------|-------------------|-------------------|
| 1 hour demo | $0.15 | $0.22 |
| 10 hours thesis work | $1.50 | $2.20 |
| 50 hours total | $7.50 | $11.00 |

Tip: **Stop the instance** when not using it — you only pay for storage when stopped (~$0.01/hr).
