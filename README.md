# Civic Crossref

Source-grounded RAG demo for querying two Scottish Borders planning documents:

- Local Development Plan 2024 (policy intent and planning framework)
- Delivery Programme 2024 (delivery and implementation detail)

The app is intentionally lightweight: FastAPI backend, in-memory cosine retrieval over a local embedding index, and a simple web UI with citations.

## Current Status

- One-shot question and answer flow (not chat yet)
- Returns answer plus cited source excerpts and page references
- Deployed target: Render + custom domain

## Architecture

1. PDFs are extracted page-by-page (PyMuPDF; optional selective Mistral OCR fallback)
2. Pages are quality-checked and normalized into a unified JSONL format
3. Text is chunked into retrieval-ready units with metadata
4. Chunks are embedded with OpenAI (`text-embedding-3-small`)
5. Runtime query flow:
   - embed user query
   - in-memory cosine similarity against all chunk vectors
   - select top-k chunks
   - generate answer with OpenAI chat model (`gpt-4o-mini`) using retrieved context
   - return answer + source list

## Repository Structure

- `app.py` - FastAPI app (`/`, `/api/health`, `/api/query`)
- `web/index.html` - client UI
- `scripts/extract_with_fallback.py` - extraction and mode selection (`text`/`blocks`/`mistral`)
- `scripts/chunk_quality_pages.py` - chunking pipeline
- `scripts/build_retrieval_index.py` - embedding index builder
- `scripts/mistral_ocr_client.py` - Mistral OCR API client helpers
- `extraction_output/index/` - runtime embedding index used by app
- `render.yaml` - Render deploy config

## Prerequisites

- Python 3.11+
- `uv` (recommended)
- OpenAI API key
- Optional: Mistral API key for selective OCR fallback during preprocessing

## Local Setup

```bash
cd /Users/martindisley/workspace/civic-crossref
uv venv .venv
uv sync
```

Create `.env`:

```bash
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional (preprocessing only)
MISTRAL_API_KEY=...
```

Run the app:

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/`.

## API Endpoints

- `GET /api/health`
  - returns service status and loaded chunk count
- `POST /api/query`
  - body: `{ "question": "...", "top_k": 5 }`
  - returns answer + source citations

Example:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does Policy HD1 cover?","top_k":5}'
```

## Preprocessing and Index Build

Only needed when rebuilding the corpus/index.

### 1) Extract with fallback

Without Mistral:

```bash
.venv/bin/python scripts/extract_with_fallback.py
```

With selective Mistral fallback:

```bash
.venv/bin/python scripts/extract_with_fallback.py --enable-mistral --mistral-max-pages-per-doc 30
```

Output (quality-checked pages):

- `extraction_output/quality_checked/*.pages.chosen.jsonl`

### 2) Chunk

```bash
.venv/bin/python scripts/chunk_quality_pages.py
```

Output (retrieval chunks):

- `extraction_output/chunks/all.chunks.retrieval.jsonl`

### 3) Build embeddings index

```bash
.venv/bin/python scripts/build_retrieval_index.py
```

Output (runtime index):

- `extraction_output/index/openai-text-embedding-3-small.chunks.jsonl`
- `extraction_output/index/openai-text-embedding-3-small.manifest.json`

## Deployment (Render)

This repo includes `render.yaml` for blueprint-based deploys.

Required env vars in Render:

- `OPENAI_API_KEY`

Optional overrides:

- `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `INDEX_PATH` (default `extraction_output/index/openai-text-embedding-3-small.chunks.jsonl`)

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Custom domain target:

- `civic-crossref.martindisley.co.uk`

## Notes and Limitations

- Current UX is one-shot QA, not multi-turn chat
- Retrieval is brute-force in-memory scan (fine for current corpus size)
- No auth layer yet (treat as collaborator prototype)
- Quality depends on retrieval + chunking; continue tuning with real queries

## Demo Script (Suggested Queries)

Use these to quickly demonstrate policy + delivery cross-referencing:

1. `What does Policy HD1 require for affordable housing delivery?`
2. `What delivery constraints are identified for allocated sites in the Delivery Programme?`
3. `How do affordable housing policy requirements compare with implementation constraints?`
4. `What does the Local Development Plan say about housing land requirements?`
5. `Which issues might affect whether LDP housing ambitions are delivered on time?`

Tip: for cross-document questions, set `top_k` to `5` or `6`.

## TODO

- Add document-aware retrieval so comparison-style queries pull relevant chunks from both Local Development Plan and Delivery Programme, not only the larger corpus.
- Add multi-turn chat mode with session-aware context, while preserving source-grounded retrieval and citations on each response.
