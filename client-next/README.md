# VisualRef — Next.js frontend

The web UI for VisualRef. Full setup (Python backend, data, models) is documented in the **[repository root README](../README.md)**.

Quick local dev (backend must already run on port 8001):

```bash
npm install
# Point at your API (local default):
echo 'NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:8001' > .env.local
npm run dev
```

Use `./start.sh` from the repo root to start backend and frontend together.
