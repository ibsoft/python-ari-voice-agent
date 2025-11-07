#!/usr/bin/env python3
"""Bulk-ingest PDF documents from docs/ into the Elasticsearch knowledge index."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml
from elasticsearch import Elasticsearch
from openai import OpenAI
from pypdf import PdfReader


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_es_client(cfg: dict) -> Elasticsearch:
    es_cfg = cfg.get("elasticsearch", {})
    hosts = es_cfg.get("hosts") or [es_cfg.get("host") or "http://localhost:9200"]
    if isinstance(hosts, str):
        hosts = [hosts]
    kwargs = {
        "hosts": hosts,
        "verify_certs": es_cfg.get("verify_certs", True),
        "request_timeout": es_cfg.get("request_timeout", 10.0),
    }
    if es_cfg.get("ca_certs"):
        kwargs["ca_certs"] = es_cfg["ca_certs"]
    if es_cfg.get("api_key"):
        kwargs["api_key"] = es_cfg["api_key"]
    elif es_cfg.get("username") and es_cfg.get("password"):
        kwargs["basic_auth"] = (es_cfg["username"], es_cfg["password"])
    return Elasticsearch(**kwargs)


def build_openai_client(cfg: dict) -> OpenAI:
    oa_cfg = cfg.get("openai", {})
    return OpenAI(
        organization=oa_cfg.get("organization"),
        project=oa_cfg.get("project")
    )


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_text(path: Path) -> List[str]:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return pages


def batched(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Elasticsearch knowledge index")
    parser.add_argument("--docs-dir", default="docs", help="Folder containing PDF files")
    parser.add_argument("--index", default=None, help="Elasticsearch index to write (defaults to knowledge index from config)")
    parser.add_argument("--chunk-chars", type=int, default=1500, help="Character length per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap characters between chunks")
    parser.add_argument("--batch", type=int, default=16, help="Embedding batch size")
    args = parser.parse_args()

    cfg = load_config()
    es = build_es_client(cfg)
    oa = build_openai_client(cfg)

    es_conf = cfg.get("elasticsearch", {})
    knowledge_index = None
    for entry in es_conf.get("indexes", []):
        if isinstance(entry, dict) and entry.get("role") == "knowledge":
            knowledge_index = entry.get("name")
            break
    index_name = args.index or knowledge_index or es_conf.get("index")
    if not index_name:
        raise SystemExit("No knowledge index configured. Use --index to specify it explicitly.")

    embedding_model = es_conf.get("embedding_model", "text-embedding-3-small")
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory {docs_dir} not found")

    pdf_files = sorted(p for p in docs_dir.rglob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found under {docs_dir}")

    for pdf in pdf_files:
        pages = extract_pdf_text(pdf)
        for page_no, page_text in enumerate(pages, start=1):
            chunks = chunk_text(page_text, args.chunk_chars, args.overlap)
            if not chunks:
                continue
            for batch in batched(chunks, args.batch):
                resp = oa.embeddings.create(model=embedding_model, input=list(batch))
                for text_chunk, emb, idx in zip(batch, resp.data, range(len(batch))):
                    payload = {
                        "content": text_chunk,
                        "file": pdf.name,
                        "path": str(pdf.relative_to(docs_dir)),
                        "page": page_no,
                        "chunk": idx,
                        "embedding": emb.embedding,
                    }
                    es.index(index=index_name, document=payload)
        print(f"Indexed {pdf}")


if __name__ == "__main__":
    main()
