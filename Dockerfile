FROM python:3.11-slim

# System packages needed by the Ollama installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces runs containers as UID 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=qwen2.5:0.5b \
    PORT=7860

WORKDIR /home/user/app

# Python dependencies.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the model into the image so cold starts don't download it.
RUN ollama serve > /tmp/ollama-build.log 2>&1 & \
    until curl -sf http://127.0.0.1:11434/api/tags > /dev/null; do sleep 1; done; \
    ollama pull "$OLLAMA_MODEL"

# Application code.
COPY --chown=user . .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
