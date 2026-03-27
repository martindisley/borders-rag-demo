#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken


INPUT_GLOB = "*.pages.chosen.jsonl"
MODEL_ENCODING = "cl100k_base"
TARGET_TOKENS = 800
OVERLAP_TOKENS = 120
MIN_CHUNK_TOKENS = 80


@dataclass
class Chunk:
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
    retrieval_exclude: bool


def token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove immediate duplicated lines commonly found in extracted contents pages.
    lines = [line.strip() for line in text.splitlines()]
    deduped: list[str] = []
    recent_norms: list[str] = []
    for line in lines:
        norm = line.lower().strip()
        if norm and norm in recent_norms[-3:]:
            continue
        deduped.append(line)
        if norm:
            recent_norms.append(norm)
    text = "\n".join(deduped)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_by_separator(text: str, sep: str) -> list[str]:
    parts = [p.strip() for p in text.split(sep)]
    return [p for p in parts if p]


def recursive_split(
    text: str, encoder: tiktoken.Encoding, target_tokens: int
) -> list[str]:
    if token_count(text, encoder) <= target_tokens:
        return [text]

    for sep in ("\n\n", "\n"):
        parts = split_by_separator(text, sep)
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                if token_count(part, encoder) <= target_tokens:
                    out.append(part)
                else:
                    out.extend(recursive_split(part, encoder, target_tokens))
            return out

    sentences = sentence_split(text)
    if len(sentences) > 1:
        out: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if token_count(candidate, encoder) <= target_tokens:
                current = candidate
            else:
                if current:
                    out.append(current)
                current = sentence
        if current:
            out.append(current)
        return out

    words = text.split()
    if not words:
        return []

    # Last resort for unstructured pages: word windows.
    out: list[str] = []
    window: list[str] = []
    for word in words:
        window.append(word)
        if token_count(" ".join(window), encoder) > target_tokens:
            if len(window) == 1:
                out.append(window[0])
                window = []
            else:
                out.append(" ".join(window[:-1]))
                window = [window[-1]]
    if window:
        out.append(" ".join(window))
    return out


def apply_overlap(
    chunks: list[str], encoder: tiktoken.Encoding, overlap_tokens: int
) -> list[str]:
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for idx in range(1, len(chunks)):
        prev = chunks[idx - 1]
        curr = chunks[idx]

        prev_tokens = encoder.encode(prev)
        overlap = (
            prev_tokens[-overlap_tokens:]
            if len(prev_tokens) > overlap_tokens
            else prev_tokens
        )
        overlap_text = encoder.decode(overlap).strip()

        if overlap_text and not curr.startswith(overlap_text):
            curr = f"{overlap_text}\n\n{curr}".strip()

        overlapped.append(curr)
    return overlapped


def is_retrieval_excluded(page_record: dict) -> bool:
    text = (page_record.get("text") or "").strip()
    if not text:
        return True

    quality = page_record.get("quality", {})
    metrics = quality.get("text_metrics", {})
    char_count = int(metrics.get("char_count", len(text)))
    short_ratio = float(metrics.get("short_line_ratio", 0.0))
    nav_ratio = float(metrics.get("nav_like_line_ratio", 0.0))

    lower = text.lower()

    looks_cover = (
        char_count < 180
        and "scottish borders council" in lower
        and ("local development plan" in lower or "delivery programme" in lower)
    )
    looks_contents = "contents" in lower and short_ratio > 0.45
    looks_contents_strong = "contents" in lower and (
        "table" in lower or "introduction" in lower or "policy" in lower
    )
    looks_nav_heavy = nav_ratio > 0.35 and short_ratio > 0.45
    looks_too_sparse = char_count < 120
    policy_hits = lower.count("policy")
    bullet_hits = lower.count("•") + lower.count("- policy")
    looks_index_like = policy_hits >= 10 and (short_ratio > 0.35 or bullet_hits >= 8)

    return (
        looks_cover
        or looks_contents
        or looks_contents_strong
        or looks_nav_heavy
        or looks_too_sparse
        or looks_index_like
    )


def page_to_chunks(
    page_record: dict,
    encoder: tiktoken.Encoding,
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
) -> list[Chunk]:
    text = normalize_whitespace(
        str(page_record.get("embed_text") or page_record.get("text") or "")
    )
    if not text:
        return []

    retrieval_exclude = is_retrieval_excluded(page_record)
    doc_id = str(page_record["doc_id"])
    page = int(page_record["page"])

    pieces = recursive_split(text, encoder, target_tokens)
    pieces = [p for p in pieces if p]
    pieces = apply_overlap(pieces, encoder, overlap_tokens)

    chunks: list[Chunk] = []
    chunk_i = 0
    for piece in pieces:
        piece = normalize_whitespace(piece)
        if not piece:
            continue

        t_count = token_count(piece, encoder)
        if t_count < min_chunk_tokens and chunks:
            merged = f"{chunks[-1].text}\n\n{piece}".strip()
            chunks[-1].text = merged
            chunks[-1].token_count = token_count(merged, encoder)
            continue

        chunk_i += 1
        chunk_id = f"{doc_id}-p{page:03d}-c{chunk_i:02d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=str(page_record["doc_title"]),
                source_file=str(page_record["source_file"]),
                page_start=page,
                page_end=page,
                chunk_index=chunk_i,
                text=piece,
                token_count=t_count,
                extraction_mode=str(page_record.get("extraction_mode", "text")),
                retrieval_exclude=retrieval_exclude,
            )
        )

    return chunks


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_dir = root / "extraction_output" / "quality_checked"
    out_dir = root / "extraction_output" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = tiktoken.get_encoding(MODEL_ENCODING)

    source_files = sorted(in_dir.glob(INPUT_GLOB))
    if not source_files:
        raise SystemExit(f"No input files matching {INPUT_GLOB} in {in_dir}")

    corpus_stats: list[dict] = []
    combined_all: list[dict] = []
    combined_retrieval: list[dict] = []

    for input_path in source_files:
        doc_all: list[dict] = []
        doc_retrieval: list[dict] = []
        pages_total = 0
        pages_excluded = 0

        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                pages_total += 1
                page_record = json.loads(line)
                chunks = page_to_chunks(
                    page_record,
                    encoder,
                    target_tokens=TARGET_TOKENS,
                    overlap_tokens=OVERLAP_TOKENS,
                    min_chunk_tokens=MIN_CHUNK_TOKENS,
                )
                if not chunks:
                    pages_excluded += 1
                    continue

                if chunks and chunks[0].retrieval_exclude:
                    pages_excluded += 1

                for chunk in chunks:
                    rec = {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "doc_title": chunk.doc_title,
                        "source_file": chunk.source_file,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "extraction_mode": chunk.extraction_mode,
                        "retrieval_exclude": chunk.retrieval_exclude,
                    }
                    doc_all.append(rec)
                    combined_all.append(rec)
                    if not chunk.retrieval_exclude:
                        doc_retrieval.append(rec)
                        combined_retrieval.append(rec)

        stem = input_path.name.replace(".pages.chosen.jsonl", "")
        all_path = out_dir / f"{stem}.chunks.all.jsonl"
        retrieval_path = out_dir / f"{stem}.chunks.retrieval.jsonl"
        write_jsonl(all_path, doc_all)
        write_jsonl(retrieval_path, doc_retrieval)

        token_counts = [r["token_count"] for r in doc_retrieval]
        avg_tokens = round(sum(token_counts) / max(1, len(token_counts)), 2)

        stats = {
            "doc": stem,
            "input": str(input_path.relative_to(root)),
            "pages_total": pages_total,
            "pages_excluded_from_retrieval": pages_excluded,
            "chunks_all": len(doc_all),
            "chunks_retrieval": len(doc_retrieval),
            "avg_tokens_retrieval": avg_tokens,
            "output_all": str(all_path.relative_to(root)),
            "output_retrieval": str(retrieval_path.relative_to(root)),
        }
        corpus_stats.append(stats)
        print(
            f"Chunked {stem}: {len(doc_retrieval)} retrieval chunks "
            f"({pages_total - pages_excluded}/{pages_total} pages included)"
        )

    write_jsonl(out_dir / "all.chunks.all.jsonl", combined_all)
    write_jsonl(out_dir / "all.chunks.retrieval.jsonl", combined_retrieval)

    stats_path = out_dir / "chunk-stats.json"
    stats_path.write_text(json.dumps(corpus_stats, indent=2), encoding="utf-8")

    report_lines = [
        "# Chunking Report",
        "",
        f"- Target tokens: {TARGET_TOKENS}",
        f"- Overlap tokens: {OVERLAP_TOKENS}",
        f"- Min chunk tokens: {MIN_CHUNK_TOKENS}",
        f"- Combined retrieval chunks: {len(combined_retrieval)}",
        "",
    ]
    for stat in corpus_stats:
        report_lines.append(f"## {stat['doc']}")
        report_lines.append(f"- Pages total: {stat['pages_total']}")
        report_lines.append(
            f"- Pages excluded from retrieval: {stat['pages_excluded_from_retrieval']}"
        )
        report_lines.append(f"- Chunks (all): {stat['chunks_all']}")
        report_lines.append(f"- Chunks (retrieval): {stat['chunks_retrieval']}")
        report_lines.append(
            f"- Avg tokens/chunk (retrieval): {stat['avg_tokens_retrieval']}"
        )
        report_lines.append(f"- Output (retrieval): `{stat['output_retrieval']}`")
        report_lines.append("")

    report_path = out_dir / "chunk-report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote stats: {stats_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
