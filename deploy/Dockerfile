FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt requests

# Install SAM 2 (public model, pulls from GitHub; weights download from HF Hub on first run)
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git"

# Copy server code
COPY server/src /app/src
COPY configs /app/../configs

# Cloud .env — SAM 2 on CUDA, Ollama on localhost
COPY deploy/env.cloud /app/.env

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "src.retrieval_server_visual:app", "--host", "0.0.0.0", "--port", "8001"]
