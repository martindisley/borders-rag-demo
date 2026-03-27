#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class IndexedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    token_count: int
    extraction_mode: str
    embedding: list[float]
    norm: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceResult(BaseModel):
    source_id: str
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    page_start: int
    page_end: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]


def post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, method="POST", headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code} {detail}") from e


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity(
    query_vec: list[float], query_norm: float, doc: IndexedChunk
) -> float:
    if query_norm == 0 or doc.norm == 0:
        return 0.0
    dot = sum(q * d for q, d in zip(query_vec, doc.embedding))
    return dot / (query_norm * doc.norm)


def truncate_excerpt(text: str, max_chars: int = 420) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class RAGService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.embedding_model = os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.chat_model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.index_path = Path(
            os.environ.get(
                "INDEX_PATH",
                "extraction_output/index/openai-text-embedding-3-small.chunks.jsonl",
            )
        )
        self.chunks: list[IndexedChunk] = []

    def validate_config(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in environment/.env")
        resolved_index = self.project_root / self.index_path
        if not resolved_index.exists():
            raise RuntimeError(f"Embedding index not found: {resolved_index}")

    def load_index(self) -> None:
        resolved_index = self.project_root / self.index_path
        chunks: list[IndexedChunk] = []
        with resolved_index.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                embedding = [float(v) for v in row["embedding"]]
                chunks.append(
                    IndexedChunk(
                        chunk_id=str(row["chunk_id"]),
                        doc_id=str(row["doc_id"]),
                        doc_title=str(row["doc_title"]),
                        source_file=str(row["source_file"]),
                        page_start=int(row["page_start"]),
                        page_end=int(row["page_end"]),
                        chunk_index=int(row["chunk_index"]),
                        text=str(row["text"]),
                        token_count=int(row.get("token_count", 0)),
                        extraction_mode=str(row.get("extraction_mode", "text")),
                        embedding=embedding,
                        norm=l2_norm(embedding),
                    )
                )
        self.chunks = chunks

    def embed_query(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": text,
        }
        data = post_json(OPENAI_EMBEDDINGS_URL, headers, payload)
        rows = data.get("data", [])
        if not rows:
            raise RuntimeError("No embedding returned for query")
        return [float(v) for v in rows[0]["embedding"]]

    def retrieve(self, question: str, top_k: int) -> list[tuple[float, IndexedChunk]]:
        q_vec = self.embed_query(question)
        q_norm = l2_norm(q_vec)

        scored: list[tuple[float, IndexedChunk]] = []
        for chunk in self.chunks:
            score = cosine_similarity(q_vec, q_norm, chunk)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def generate_answer(
        self, question: str, ranked_chunks: list[tuple[float, IndexedChunk]]
    ) -> str:
        context_parts: list[str] = []
        for i, (score, chunk) in enumerate(ranked_chunks, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"[S{i}] doc={chunk.doc_title} file={chunk.source_file} pages={chunk.page_start}-{chunk.page_end} score={score:.4f}",
                        chunk.text,
                    ]
                )
            )
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You answer questions using two Scottish Borders planning documents: "
            "(1) Local Development Plan 2024 (policy framework, strategic aims, and planning policies), and "
            "(2) Delivery Programme 2024 (implementation status, actions, and delivery/monitoring detail). "
            "Use only the provided sources. If the answer is not in sources, say so. "
            "When relevant, distinguish policy intent (LDP) from delivery evidence (Delivery Programme). "
            "When sources from both documents are available, cross-reference both and cite each explicitly. "
            "If only one document is represented in retrieved sources, state that limitation clearly. "
            "Cite sources inline as [S1], [S2], etc. Keep the answer concise, factual, and explicit about uncertainty."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Sources:\n{context}\n\n"
            "Provide a direct answer with inline source tags. "
            "If possible, include one brief sentence that compares policy intent vs delivery status."
        )

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chat_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = post_json(OPENAI_CHAT_URL, headers, payload)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No completion choices returned")
        return str(choices[0]["message"]["content"]).strip()

    def query(self, question: str, top_k: int) -> QueryResponse:
        ranked = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, ranked)

        sources: list[SourceResult] = []
        for i, (score, chunk) in enumerate(ranked, start=1):
            sources.append(
                SourceResult(
                    source_id=f"S{i}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    source_file=chunk.source_file,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=round(score, 6),
                    excerpt=truncate_excerpt(chunk.text),
                )
            )

        return QueryResponse(answer=answer, sources=sources)


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent
WEB_INDEX = PROJECT_ROOT / "web" / "index.html"
rag_service = RAGService(PROJECT_ROOT)

app = FastAPI(title="Borders RAG Demo API", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    rag_service.validate_config()
    rag_service.load_index()


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "chunks_loaded": len(rag_service.chunks),
        "embedding_model": rag_service.embedding_model,
        "chat_model": rag_service.chat_model,
    }


@app.get("/")
def web_app() -> FileResponse:
    if not WEB_INDEX.exists():
        raise HTTPException(status_code=404, detail="Web UI not found")
    return FileResponse(WEB_INDEX)


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        return rag_service.query(req.question, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
