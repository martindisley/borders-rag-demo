#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embedding index from retrieval chunks"
    )
    parser.add_argument(
        "--input",
        default="extraction_output/chunks/all.chunks.retrieval.jsonl",
        help="Input chunk JSONL path (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        default="extraction_output/index",
        help="Output directory (relative to project root)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embeddings model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between API calls",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method="POST", headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code} {detail}") from e


def embed_batch(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": texts,
    }
    data = post_json(OPENAI_EMBEDDINGS_URL, headers, payload)
    rows = data.get("data", [])
    if len(rows) != len(texts):
        raise RuntimeError(
            f"Embeddings response size mismatch: got {len(rows)} expected {len(texts)}"
        )
    return [row["embedding"] for row in rows]


def batched(items: list[dict], size: int) -> list[list[dict]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    load_dotenv()
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing in environment/.env")

    chunks = load_chunks(input_path)
    if not chunks:
        raise SystemExit(f"No chunks found in {input_path}")

    all_batches = batched(chunks, args.batch_size)

    output_jsonl = out_dir / f"openai-{args.model}.chunks.jsonl"
    manifest_path = out_dir / f"openai-{args.model}.manifest.json"

    total = len(chunks)
    done = 0
    embedding_dim = None

    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, batch in enumerate(all_batches, start=1):
            texts = [str(c.get("text", "")) for c in batch]
            vectors = embed_batch(texts, args.model, api_key)

            for chunk, vec in zip(batch, vectors):
                if embedding_dim is None:
                    embedding_dim = len(vec)

                rec = {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "doc_title": chunk.get("doc_title"),
                    "source_file": chunk.get("source_file"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "chunk_index": chunk.get("chunk_index"),
                    "text": chunk.get("text"),
                    "token_count": chunk.get("token_count"),
                    "extraction_mode": chunk.get("extraction_mode"),
                    "embedding": vec,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            done += len(batch)
            print(f"Embedded batch {i}/{len(all_batches)} ({done}/{total})")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    manifest = {
        "input": str(input_path.relative_to(root)),
        "output": str(output_jsonl.relative_to(root)),
        "model": args.model,
        "batch_size": args.batch_size,
        "chunks": total,
        "embedding_dimensions": embedding_dim,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote index: {output_jsonl}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
