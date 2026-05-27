# Agent Notes

## Setup And Runtime
- This project requires Python 3.12 and is managed with `uv`.
- Install runtime deps with `uv sync`; Render uses `pip install uv && uv sync --frozen`.
- Run locally with `uv run python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload`.
- App startup fails unless `OPENAI_API_KEY` is present and `INDEX_PATH` exists; the default index is `extraction_output/index/openai-text-embedding-3-small.chunks.jsonl`.
- Test command is `uv run --group dev pytest`.
- There are no configured lint, format, or typecheck commands in this repo. A cheap syntax check is `uv run python -m py_compile app.py scripts/*.py tests/*.py`.

## Architecture
- `app.py` is the whole FastAPI backend and serves `web/index.html` at `/`; API endpoints are `/api/health` and `/api/query`.
- `app.py` now exposes `create_app(service=...)` for tests; keep module-level `app = create_app()` for `uvicorn app:app`.
- Retrieval is an in-memory cosine scan over the JSONL embedding index loaded at startup, then `/api/query` calls OpenAI chat completions.
- The web UI is a single static HTML/CSS/JS file; it calls the backend with relative `/api/*` URLs.
- The checked-in index manifest currently records 278 chunks using `text-embedding-3-small` with 1536-dimensional embeddings.

## Corpus Pipeline
- Rebuild order is fixed: extract pages, chunk pages, then build embeddings.
- Commands are `uv run python scripts/extract_with_fallback.py`, `uv run python scripts/chunk_quality_pages.py`, then `uv run python scripts/build_retrieval_index.py`.
- `scripts/extract_with_fallback.py` expects ignored local PDFs at repo root: `local-development-plan.pdf` and `delivery-programme.pdf`.
- Optional OCR is `uv run python scripts/extract_with_fallback.py --enable-mistral --mistral-max-pages-per-doc 30` and also needs `MISTRAL_API_KEY`.
- `pyproject.toml` only declares runtime deps; preprocessing imports `fitz`/PyMuPDF and `tiktoken`, so do not assume `uv sync` alone is enough to rerun the corpus pipeline until those deps are added or installed.

## Files To Preserve
- `.gitignore` intentionally ignores raw PDFs and most `extraction_output/*`; only the runtime index JSONL and manifest are unignored for deployment.
- If changing embedding model names, update both generated index filenames and the app/Render `INDEX_PATH` default so startup still finds the index.
