#!/usr/bin/env bash
set -e

# Start Ollama in the background.
ollama serve &

# Wait until Ollama is ready.
echo "Waiting for Ollama..."
until curl -sf http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready."

# Warm up the model so the first request is fast (non-fatal).
curl -sf http://127.0.0.1:11434/api/generate \
    -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"hi\", \"stream\": false}" \
    > /dev/null || true

# Start the FastAPI app.
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
