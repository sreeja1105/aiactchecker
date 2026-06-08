"""
Document ingestion pipeline.
PDF -> text -> chunks -> embeddings -> ChromaDB.
"""

import time
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

from src.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.llm import embed_batch


def extract_pdf_text(pdf_path: Path) -> List[Dict]:
    """Extract text from each PDF page, preserving page numbers."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page": i})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Breaks at paragraph or sentence boundaries when possible.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        break_point = text.rfind("\n\n", start + chunk_size // 2, end)
        if break_point == -1:
            break_point = text.rfind(". ", start + chunk_size // 2, end)
            if break_point != -1:
                break_point += 2
        if break_point == -1:
            break_point = end

        chunk = text[start:break_point].strip()
        if chunk:
            chunks.append(chunk)

        new_start = break_point - overlap
        start = new_start if new_start > start else break_point

    return chunks


def get_chroma_collection():
    """Open or create the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "EU AI Act and related compliance documents"},
    )


def ingest_pdf(pdf_path: Path, source_name: str = None) -> None:
    """Ingest a PDF: extract text, chunk, embed, store in ChromaDB."""
    if source_name is None:
        source_name = pdf_path.stem

    print(f"[ingest] {pdf_path.name}")
    print(f"  source label: {source_name}")

    pages = extract_pdf_text(pdf_path)
    print(f"  pages extracted: {len(pages)}")

    documents, metadatas, ids = [], [], []
    chunk_id = 0
    for page_data in pages:
        for chunk in chunk_text(page_data["text"]):
            if len(chunk) < 50:
                continue
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "page": page_data["page"],
                "file": pdf_path.name,
            })
            ids.append(f"{source_name}_chunk_{chunk_id}")
            chunk_id += 1

    print(f"  chunks prepared: {len(documents)}")
    print(f"  embedding (local sentence-transformers)...")

    start = time.time()
    embeddings = embed_batch(documents)
    elapsed = time.time() - start
    print(f"  embedded {len(documents)} chunks in {elapsed:.1f}s ({len(documents)/elapsed:.0f}/sec)")

    collection = get_chroma_collection()
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )
    print(f"  stored in ChromaDB. collection size: {collection.count()}")
