# Customer Support RAG API

FastAPI + Retrieval-Augmented Generation. Uses a local Ollama model (`qwen2.5:0.5b`) in development and the hosted Groq API (`llama-3.1-8b-instant`) when deployed.

## Deploy on Render (free)

1. Get a free API key at https://console.groq.com/keys
2. Render dashboard → **New → Blueprint** → select this repo (uses `render.yaml`).
3. Paste your `GROQ_API_KEY` when prompted → **Apply**.

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
| `LLM_PROVIDER`        | `ollama` (or `groq`)     |
| `GROQ_API_KEY`        | *(required for groq)*    |
| `GROQ_MODEL`          | `llama-3.1-8b-instant`   |
| `OLLAMA_BASE_URL`     | `http://localhost:11434` |
| `OLLAMA_MODEL`        | `qwen2.5:0.5b`           |
| `MAX_QUESTION_LENGTH` | `1000`                   |
