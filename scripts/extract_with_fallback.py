#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import median

import fitz  # PyMuPDF
from dotenv import load_dotenv

from mistral_ocr_client import fetch_mistral_pages, markdown_to_embed_text


DOCS = [
    ("ldp", "Local Development Plan", "local-development-plan.pdf"),
    ("delivery_programme", "Delivery Programme 2024", "delivery-programme.pdf"),
]


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def blocks_to_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks", sort=True)
    ordered: list[str] = []
    for block in blocks:
        text = (block[4] or "").strip()
        if text:
            ordered.append(text)
    return "\n\n".join(ordered)


def normalize_line(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\d+", "", line)
    line = re.sub(r"[^a-z| ]", "", line)
    return line.strip()


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def line_word_count(line: str) -> int:
    return len(re.findall(r"\b\w+\b", line))


def metrics_for_text(text: str) -> dict[str, float | int]:
    lines = nonempty_lines(text)
    words = re.findall(r"\b\w+\b", text)

    short_lines = 0
    uppercase_lines = 0
    nav_like_lines = 0
    punct_end_lines = 0

    for line in lines:
        wc = line_word_count(line)
        if wc <= 3:
            short_lines += 1

        letters = [c for c in line if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.9:
                uppercase_lines += 1

        if "|" in line or "contents" in line.lower():
            nav_like_lines += 1

        if line.endswith((".", ":", ";", "?", "!")):
            punct_end_lines += 1

    alpha_chars = sum(1 for c in text if c.isalpha())
    non_ws_chars = sum(1 for c in text if not c.isspace())

    line_count = len(lines)
    word_count = len(words)

    return {
        "char_count": len(text),
        "line_count": line_count,
        "word_count": word_count,
        "avg_words_per_line": round(word_count / max(1, line_count), 3),
        "short_line_ratio": round(short_lines / max(1, line_count), 3),
        "uppercase_line_ratio": round(uppercase_lines / max(1, line_count), 3),
        "nav_like_line_ratio": round(nav_like_lines / max(1, line_count), 3),
        "punct_end_line_ratio": round(punct_end_lines / max(1, line_count), 3),
        "alpha_ratio": round(alpha_chars / max(1, non_ws_chars), 3),
    }


def trim_repeated_edges(text: str, repeated_lines: set[str]) -> str:
    lines = nonempty_lines(text)
    if not lines:
        return ""

    while lines and normalize_line(lines[0]) in repeated_lines:
        lines.pop(0)
    while lines and normalize_line(lines[-1]) in repeated_lines:
        lines.pop()

    return clean_text("\n".join(lines))


def score_page(
    text_metrics: dict[str, float | int],
    blocks_metrics: dict[str, float | int],
    median_chars: float,
    repeated_edge_hits: int,
) -> tuple[int, int, list[str], list[str]]:
    badness = 0
    benefit = 0
    bad_reasons: list[str] = []
    benefit_reasons: list[str] = []

    t_chars = int(text_metrics["char_count"])
    t_words = int(text_metrics["word_count"])
    t_lines = int(text_metrics["line_count"])
    t_avg_wpl = float(text_metrics["avg_words_per_line"])
    t_short = float(text_metrics["short_line_ratio"])
    t_upper = float(text_metrics["uppercase_line_ratio"])
    t_nav = float(text_metrics["nav_like_line_ratio"])
    t_alpha = float(text_metrics["alpha_ratio"])

    b_chars = int(blocks_metrics["char_count"])
    b_avg_wpl = float(blocks_metrics["avg_words_per_line"])
    b_short = float(blocks_metrics["short_line_ratio"])
    b_alpha = float(blocks_metrics["alpha_ratio"])

    if t_chars < (0.35 * median_chars) and t_words < 80:
        badness += 2
        bad_reasons.append("very low text volume")
    if t_short > 0.55:
        badness += 2
        bad_reasons.append("high short-line ratio")
    if t_upper > 0.45:
        badness += 1
        bad_reasons.append("high uppercase-line ratio")
    if t_nav > 0.2:
        badness += 1
        bad_reasons.append("nav/header style lines")
    if t_alpha < 0.6:
        badness += 1
        bad_reasons.append("low alphabetic ratio")
    if t_lines > 20 and t_avg_wpl < 3.0:
        badness += 1
        bad_reasons.append("fragmented line structure")
    if repeated_edge_hits >= 2:
        badness += 1
        bad_reasons.append("repeated header/footer")

    if b_chars > (1.25 * max(1, t_chars)):
        benefit += 2
        benefit_reasons.append("blocks has much more text")
    if b_short < (t_short - 0.15):
        benefit += 2
        benefit_reasons.append("blocks has fewer short fragments")
    if b_avg_wpl > (t_avg_wpl + 1.5):
        benefit += 1
        benefit_reasons.append("blocks has better line coherence")
    if b_alpha > (t_alpha + 0.05):
        benefit += 1
        benefit_reasons.append("blocks has cleaner character mix")

    return badness, benefit, bad_reasons, benefit_reasons


def mistral_candidate(page_metrics: dict[str, float | int], badness: int) -> bool:
    short_ratio = float(page_metrics["short_line_ratio"])
    avg_wpl = float(page_metrics["avg_words_per_line"])
    nav_ratio = float(page_metrics["nav_like_line_ratio"])
    line_count = int(page_metrics["line_count"])

    return (
        badness >= 4
        or (short_ratio > 0.6 and line_count > 20)
        or (avg_wpl < 2.8 and line_count > 25)
        or nav_ratio > 0.35
    )


def mistral_benefit_score(
    selected_metrics: dict[str, float | int], mistral_metrics: dict[str, float | int]
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    sel_short = float(selected_metrics["short_line_ratio"])
    sel_avg = float(selected_metrics["avg_words_per_line"])
    sel_chars = int(selected_metrics["char_count"])
    sel_alpha = float(selected_metrics["alpha_ratio"])

    mis_short = float(mistral_metrics["short_line_ratio"])
    mis_avg = float(mistral_metrics["avg_words_per_line"])
    mis_chars = int(mistral_metrics["char_count"])
    mis_alpha = float(mistral_metrics["alpha_ratio"])

    if mis_short < (sel_short - 0.15):
        score += 2
        reasons.append("mistral has fewer short fragments")
    if mis_avg > (sel_avg + 1.2):
        score += 1
        reasons.append("mistral has better line coherence")
    if mis_chars > (sel_chars * 1.1):
        score += 1
        reasons.append("mistral captures more text")
    if mis_alpha > (sel_alpha + 0.03):
        score += 1
        reasons.append("mistral cleaner character mix")

    return score, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction with text/blocks/Mistral fallback"
    )
    parser.add_argument(
        "--enable-mistral",
        action="store_true",
        help="Enable selective Mistral OCR for problematic pages",
    )
    parser.add_argument(
        "--mistral-max-pages-per-doc",
        type=int,
        default=30,
        help="Max pages per document to OCR with Mistral in one run",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "extraction_output" / "quality_checked"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "extraction_output" / "mistral_cache"

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    use_mistral = bool(args.enable_mistral and api_key)
    if args.enable_mistral and not api_key:
        print("MISTRAL_API_KEY missing; continuing without Mistral OCR")

    all_stats: list[dict[str, object]] = []
    report_lines = ["# Extraction Fallback Report", ""]

    for doc_id, doc_title, filename in DOCS:
        pdf_path = root / filename
        if not pdf_path.exists():
            print(f"Skipping missing file: {filename}")
            continue

        doc = fitz.open(pdf_path)
        pages: list[dict[str, object]] = []
        text_char_counts: list[int] = []
        edge_counter: Counter[str] = Counter()

        for page_num, page in enumerate(doc, start=1):
            text_raw = clean_text(page.get_text("text", sort=True))
            blocks_raw = clean_text(blocks_to_text(page))

            t_metrics = metrics_for_text(text_raw)
            b_metrics = metrics_for_text(blocks_raw)

            text_lines = nonempty_lines(text_raw)
            first_norm = normalize_line(text_lines[0]) if text_lines else ""
            last_norm = normalize_line(text_lines[-1]) if text_lines else ""

            if first_norm:
                edge_counter[first_norm] += 1
            if last_norm:
                edge_counter[last_norm] += 1

            text_char_counts.append(int(t_metrics["char_count"]))

            pages.append(
                {
                    "page": page_num,
                    "text_raw": text_raw,
                    "blocks_raw": blocks_raw,
                    "text_metrics": t_metrics,
                    "blocks_metrics": b_metrics,
                    "first_norm": first_norm,
                    "last_norm": last_norm,
                }
            )

        min_repeat = max(4, int(doc.page_count * 0.06))
        repeated_edges = {
            line for line, count in edge_counter.items() if count >= min_repeat
        }
        median_chars = float(median(text_char_counts)) if text_char_counts else 0.0

        # Pre-score pages and collect candidates for selective Mistral OCR.
        candidate_pages: list[tuple[int, int]] = []
        for page in pages:
            t_metrics = dict(page["text_metrics"])
            b_metrics = dict(page["blocks_metrics"])
            repeated_hits = int(page["first_norm"] in repeated_edges) + int(
                page["last_norm"] in repeated_edges
            )
            badness, benefit, _, _ = score_page(
                t_metrics, b_metrics, median_chars, repeated_hits
            )
            page["badness"] = badness
            page["blocks_benefit"] = benefit
            if mistral_candidate(t_metrics, badness):
                candidate_pages.append((int(page["page"]), badness))

        candidate_pages.sort(key=lambda x: x[1], reverse=True)
        selected_mistral_pages = [
            p for p, _ in candidate_pages[: args.mistral_max_pages_per_doc]
        ]

        mistral_map: dict[int, dict[str, str]] = {}
        mistral_used_candidates = 0
        if use_mistral and selected_mistral_pages:
            cache_path = cache_dir / f"{doc_id}.pages.json"
            mistral_map = fetch_mistral_pages(
                pdf_path, selected_mistral_pages, cache_path, api_key
            )
            mistral_used_candidates = len(mistral_map)

        jsonl_path = out_dir / f"{doc_id}.pages.chosen.jsonl"
        review_path = out_dir / f"{doc_id}.flagged-for-review.json"

        flagged: list[dict[str, object]] = []
        chosen_blocks_count = 0
        chosen_mistral_count = 0

        with jsonl_path.open("w", encoding="utf-8") as f:
            for page in pages:
                page_num = int(page["page"])
                text_raw = str(page["text_raw"])
                blocks_raw = str(page["blocks_raw"])
                t_metrics = dict(page["text_metrics"])
                b_metrics = dict(page["blocks_metrics"])
                first_norm = str(page["first_norm"])
                last_norm = str(page["last_norm"])

                repeated_hits = int(first_norm in repeated_edges) + int(
                    last_norm in repeated_edges
                )
                badness, benefit, bad_reasons, benefit_reasons = score_page(
                    t_metrics, b_metrics, median_chars, repeated_hits
                )

                use_blocks = (badness >= 4 and benefit >= 2) or (benefit >= 5)
                needs_review = (badness >= 3 and benefit >= 1) or (
                    badness >= 5 and benefit == 0
                )

                selected_raw = blocks_raw if use_blocks else text_raw
                selected_text = trim_repeated_edges(selected_raw, repeated_edges)
                selected_metrics = b_metrics if use_blocks else t_metrics
                mode = "blocks" if use_blocks else "text"
                content_format = "plain_text"
                raw_text = selected_raw
                mistral_reasons: list[str] = []

                if use_blocks:
                    chosen_blocks_count += 1

                mistral_entry = mistral_map.get(page_num)
                if mistral_entry:
                    mistral_markdown = str(mistral_entry.get("markdown", "")).strip()
                    mistral_embed = clean_text(
                        markdown_to_embed_text(
                            str(mistral_entry.get("embed_text", mistral_markdown))
                        )
                    )
                    if mistral_embed:
                        mistral_metrics = metrics_for_text(mistral_embed)
                        m_score, mistral_reasons = mistral_benefit_score(
                            selected_metrics, mistral_metrics
                        )
                        if m_score >= 2:
                            mode = "mistral"
                            content_format = "markdown"
                            raw_text = mistral_markdown
                            selected_text = mistral_embed
                            selected_metrics = mistral_metrics
                            chosen_mistral_count += 1

                record = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "source_file": filename,
                    "page": page_num,
                    "text": selected_text,
                    "embed_text": selected_text,
                    "raw_text": raw_text,
                    "content_format": content_format,
                    "extraction_mode": mode,
                    "quality": {
                        "badness": badness,
                        "blocks_benefit": benefit,
                        "needs_review": needs_review,
                        "bad_reasons": bad_reasons,
                        "blocks_benefit_reasons": benefit_reasons,
                        "mistral_benefit_reasons": mistral_reasons,
                        "text_metrics": t_metrics,
                        "blocks_metrics": b_metrics,
                        "selected_metrics": selected_metrics,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if needs_review:
                    flagged.append(
                        {
                            "page": page_num,
                            "chosen_mode": mode,
                            "badness": badness,
                            "blocks_benefit": benefit,
                            "bad_reasons": bad_reasons,
                            "blocks_benefit_reasons": benefit_reasons,
                            "mistral_benefit_reasons": mistral_reasons,
                            "text_chars": t_metrics["char_count"],
                            "blocks_chars": b_metrics["char_count"],
                        }
                    )

        review_path.write_text(json.dumps(flagged, indent=2), encoding="utf-8")

        doc_stats = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "source_file": filename,
            "pages": doc.page_count,
            "median_text_chars": round(median_chars, 2),
            "repeated_edge_lines": len(repeated_edges),
            "mistral_enabled": use_mistral,
            "mistral_candidate_pages": len(selected_mistral_pages),
            "mistral_pages_fetched": mistral_used_candidates,
            "pages_switched_to_blocks": chosen_blocks_count,
            "pages_switched_to_mistral": chosen_mistral_count,
            "pages_flagged_for_review": len(flagged),
            "output_jsonl": str(jsonl_path.relative_to(root)),
            "flagged_review_json": str(review_path.relative_to(root)),
        }
        all_stats.append(doc_stats)

        report_lines.append(f"## {doc_title} ({filename})")
        report_lines.append(f"- Pages: {doc.page_count}")
        report_lines.append(f"- Median text chars/page: {round(median_chars, 2)}")
        report_lines.append(f"- Repeated edge lines detected: {len(repeated_edges)}")
        report_lines.append(f"- Mistral enabled: {use_mistral}")
        report_lines.append(f"- Mistral candidate pages: {len(selected_mistral_pages)}")
        report_lines.append(
            f"- Mistral pages fetched from API/cache: {mistral_used_candidates}"
        )
        report_lines.append(f"- Pages switched to blocks: {chosen_blocks_count}")
        report_lines.append(f"- Pages switched to mistral: {chosen_mistral_count}")
        report_lines.append(f"- Pages flagged for review: {len(flagged)}")
        report_lines.append(f"- Output: `{jsonl_path.relative_to(root)}`")
        report_lines.append(f"- Review list: `{review_path.relative_to(root)}`")
        report_lines.append("")

        print(
            f"Processed {filename}: {doc.page_count} pages, "
            f"blocks={chosen_blocks_count}, mistral={chosen_mistral_count}, "
            f"flagged={len(flagged)}"
        )

    stats_path = out_dir / "corpus-fallback-stats.json"
    report_path = out_dir / "fallback-report.md"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote stats: {stats_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
