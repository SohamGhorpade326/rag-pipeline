---
title: Customer Support RAG API
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support RAG API

FastAPI + Retrieval-Augmented Generation using a local Ollama model (`qwen2.5:0.5b`).

## Endpoints

| Method | Path      | Description                      |
|--------|-----------|----------------------------------|
| GET    | `/`       | Redirects to interactive docs    |
| GET    | `/health` | Health check                     |
| POST   | `/ask`    | Ask a customer support question  |
| GET    | `/docs`   | Swagger UI                       |

Example:

```bash
curl -X POST https://<your-space>.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
ollama pull qwen2.5:0.5b
uvicorn app:app --reload
```

## Run with Docker

```bash
docker build -t support-rag .
docker run -p 7860:7860 support-rag
```

## Configuration (env vars)

| Variable              | Default                  |
|-----------------------|--------------------------|
| `OLLAMA_BASE_URL`     | `http://localhost:11434` |
| `OLLAMA_MODEL`        | `qwen2.5:0.5b`           |
| `MAX_QUESTION_LENGTH` | `1000`                   |
