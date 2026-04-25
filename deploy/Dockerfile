FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app/server

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt requests

# Install SAM3 from the local clone
COPY server/sam3 /app/server/sam3
RUN cd /app/server/sam3 && pip install --no-cache-dir -e .

# Copy repo layout expected by server/src/config.py
COPY server/src /app/server/src
COPY configs /app/configs

# Cloud .env — SAM3 on CUDA, Ollama on localhost
COPY deploy/env.cloud /app/server/.env

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "src.retrieval_server_visual:app", "--host", "0.0.0.0", "--port", "8001"]
