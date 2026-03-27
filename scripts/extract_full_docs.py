#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


DOCS = [
    ("ldp", "Local Development Plan", "local-development-plan.pdf"),
    ("delivery_programme", "Delivery Programme 2024", "delivery-programme.pdf"),
]


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "extraction_output" / "full_pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_stats = []

    for doc_id, doc_title, filename in DOCS:
        pdf_path = root / filename
        if not pdf_path.exists():
            print(f"Skipping missing file: {filename}")
            continue

        doc = fitz.open(pdf_path)
        jsonl_path = out_dir / f"{doc_id}.pages.jsonl"
        txt_path = out_dir / f"{doc_id}.pages.txt"

        total_chars = 0
        records = []
        txt_parts = []

        for i, page in enumerate(doc, start=1):
            raw_text = page.get_text("text", sort=True)
            text = clean_text(raw_text)
            total_chars += len(text)

            record = {
                "doc_id": doc_id,
                "doc_title": doc_title,
                "source_file": filename,
                "page": i,
                "text": text,
            }
            records.append(record)

            txt_parts.append(f"\n\n===== {doc_id} | page {i} =====\n\n{text}\n")

        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        txt_path.write_text("".join(txt_parts), encoding="utf-8")

        stats = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "source_file": filename,
            "pages": doc.page_count,
            "total_chars": total_chars,
            "avg_chars_per_page": round(total_chars / max(doc.page_count, 1), 2),
            "jsonl": str(jsonl_path.relative_to(root)),
            "txt": str(txt_path.relative_to(root)),
        }
        corpus_stats.append(stats)
        print(f"Extracted {filename}: {doc.page_count} pages, {total_chars} chars")

    stats_path = out_dir / "corpus-stats.json"
    stats_path.write_text(json.dumps(corpus_stats, indent=2), encoding="utf-8")
    print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
