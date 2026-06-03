# configs/

YAML configuration files consumed by the indexer scripts and the runtime server.

## Layout

- `demo/` — demo-corpus configs:
  - `vg_siglip.yaml` — Visual Genome corpus (~108k images), SigLIP large/256 backbone (the active demo config).

Each config specifies: corpus path, image-size, model identifier, output FAISS path, and any encoder-specific options.
