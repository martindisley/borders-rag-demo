#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import fitz  # PyMuPDF


def blocks_to_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks", sort=True)
    ordered = []
    for block in blocks:
        text = (block[4] or "").strip()
        if text:
            ordered.append(text)
    return "\n\n".join(ordered)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "extraction_output"
    out_dir.mkdir(exist_ok=True)

    test_pdfs = sorted(root.glob("ldp-test-page-*.pdf"))
    if not test_pdfs:
        raise SystemExit("No test PDFs found matching ldp-test-page-*.pdf")

    report = ["# Extraction Quality Report (Test Pages)", ""]
    summary = []

    for pdf_path in test_pdfs:
        doc = fitz.open(pdf_path)
        report.append(f"## {pdf_path.name}")
        report.append(f"- Pages: {doc.page_count}")

        page_records = []
        for i, page in enumerate(doc, start=1):
            text_mode = page.get_text("text", sort=True).strip()
            blocks_mode = blocks_to_text(page)

            text_file = out_dir / f"{pdf_path.stem}-p{i:03d}.text.txt"
            blocks_file = out_dir / f"{pdf_path.stem}-p{i:03d}.blocks.txt"
            text_file.write_text(text_mode + "\n", encoding="utf-8")
            blocks_file.write_text(blocks_mode + "\n", encoding="utf-8")

            text_snippet = (
                (text_mode[:900] + "...") if len(text_mode) > 900 else text_mode
            )
            blocks_snippet = (
                (blocks_mode[:900] + "...") if len(blocks_mode) > 900 else blocks_mode
            )

            report.append(f"### Page {i}")
            report.append(f"- text chars: {len(text_mode)}")
            report.append(f"- blocks chars: {len(blocks_mode)}")
            report.append("")
            report.append("**text mode snippet**")
            report.append("")
            report.append("```")
            report.append(text_snippet)
            report.append("```")
            report.append("")
            report.append("**blocks mode snippet**")
            report.append("")
            report.append("```")
            report.append(blocks_snippet)
            report.append("```")
            report.append("")

            page_records.append(
                {
                    "page": i,
                    "text_chars": len(text_mode),
                    "blocks_chars": len(blocks_mode),
                    "text_file": str(text_file.relative_to(root)),
                    "blocks_file": str(blocks_file.relative_to(root)),
                }
            )

        summary.append(
            {
                "file": pdf_path.name,
                "page_count": doc.page_count,
                "pages": page_records,
            }
        )
        report.append("")

    (out_dir / "test-pages-report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "test-pages-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Wrote report: {out_dir / 'test-pages-report.md'}")
    print(f"Wrote summary: {out_dir / 'test-pages-summary.json'}")


if __name__ == "__main__":
    main()
