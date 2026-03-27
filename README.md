# Borders RAG Demo

FastAPI demo for querying Scottish Borders planning documents with source-grounded answers.

## Local Run

```bash
uv venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/`.

## Environment

Create `.env` with:

```bash
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

The app reads embeddings from:

`extraction_output/index/openai-text-embedding-3-small.chunks.jsonl`

## Deploy (Render)

- `render.yaml` is included for blueprint-based deploys
- Set `OPENAI_API_KEY` in Render dashboard
- After deploy, attach custom domain: `borders-rag-demo.martindisley.co.uk`
