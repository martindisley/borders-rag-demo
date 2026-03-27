#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path


API_BASE = "https://api.mistral.ai/v1"


def build_multipart(
    file_path: str, filename: str, mime_type: str, boundary: str
) -> bytes:
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    lines: list[str] = []
    lines.append(f"--{boundary}\r\n")
    lines.append('Content-Disposition: form-data; name="purpose"\r\n\r\n')
    lines.append("ocr\r\n")
    lines.append(f"--{boundary}\r\n")
    lines.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
    )
    lines.append(f"Content-Type: {mime_type}\r\n\r\n")
    body_start = "".join(lines).encode("utf-8")
    body_end = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body_start + file_bytes + body_end


def http_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    req = urllib.request.Request(url, method="POST", headers=headers, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def upload_file(file_path: str, api_key: str) -> str:
    filename = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(filename)[0] or "application/pdf"
    boundary = uuid.uuid4().hex
    body = build_multipart(file_path, filename, mime_type, boundary)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    status, text = http_post(f"{API_BASE}/files", headers, body)
    if status < 200 or status >= 300:
        raise RuntimeError(f"File upload failed: HTTP {status} {text}")

    data = json.loads(text)
    file_id = data.get("id")
    if not file_id:
        raise RuntimeError("File upload response missing file id")
    return file_id


def ocr_file(file_id: str, api_key: str, pages: list[int] | None) -> dict:
    payload: dict[str, object] = {
        "model": "mistral-ocr-latest",
        "document": {"type": "file", "file_id": file_id},
    }
    if pages is not None:
        payload["pages"] = pages

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    status, text = http_post(f"{API_BASE}/ocr", headers, body)
    if status < 200 or status >= 300:
        raise RuntimeError(f"OCR failed: HTTP {status} {text}")
    return json.loads(text)


def strip_markdown_images(markdown: str) -> str:
    return re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", markdown)


def markdown_to_embed_text(markdown: str) -> str:
    text = strip_markdown_images(markdown)
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_mistral_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_mistral_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def fetch_mistral_pages(
    pdf_path: Path,
    page_numbers_1_indexed: list[int],
    cache_path: Path,
    api_key: str,
) -> dict[int, dict[str, str]]:
    """Returns mapping page_number -> {markdown, embed_text}."""
    if not page_numbers_1_indexed:
        return {}

    cache = load_mistral_cache(cache_path)
    needed_pages = [
        p for p in sorted(set(page_numbers_1_indexed)) if str(p) not in cache
    ]

    if needed_pages:
        page_indexes = [p - 1 for p in needed_pages]
        file_id = upload_file(str(pdf_path), api_key)
        result = ocr_file(file_id, api_key, page_indexes)

        for page in result.get("pages", []):
            index = int(page.get("index", 0))
            page_num = index + 1
            markdown = str(page.get("markdown", "")).strip()
            cache[str(page_num)] = {
                "markdown": markdown,
                "embed_text": markdown_to_embed_text(markdown),
            }

        save_mistral_cache(cache_path, cache)

    out: dict[int, dict[str, str]] = {}
    for page_num in page_numbers_1_indexed:
        entry = cache.get(str(page_num))
        if entry:
            out[page_num] = {
                "markdown": str(entry.get("markdown", "")),
                "embed_text": str(entry.get("embed_text", "")),
            }
    return out
