# Use gonka's custom vLLM image with Blackwell support and PoC endpoints
FROM ghcr.io/gonka-ai/vllm:v0.9.1-blackwell

# Set working directory
WORKDIR /app

# Install RunPod SDK and HTTP client
# Use vLLM's Python directly
RUN /usr/bin/python3.12 -m pip install --no-cache-dir runpod requests httpx

# Copy handler code
COPY handler.py /app/
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# PoC v2 settings
ENV POC_VERSION=v2
ENV MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
ENV K_DIM=12
ENV SEQ_LEN=256

# vLLM settings
ENV VLLM_PORT=8000
ENV VLLM_HOST=127.0.0.1

# HuggingFace cache location (RunPod caches models here)
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV VLLM_USE_V1=0

# Clear any default entrypoint from vLLM image
ENTRYPOINT []

# Run startup script which starts vLLM and handler
CMD ["/app/startup.sh"]
