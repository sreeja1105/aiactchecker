"""
Retrieval pipeline.
Query -> embedding -> similarity search -> top-K chunks with citations.
"""

from typing import List, Dict

from src.ingestion import get_chroma_collection
from src.llm import embed, generate


def retrieve(query: str, k: int = 5) -> List[Dict]:
    """Retrieve top-K most relevant chunks for a query."""
    collection = get_chroma_collection()
    query_embedding = embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "score": round(1 - dist, 3),
        })
    return hits


def answer_with_citations(query: str, k: int = 5) -> Dict:
    """Retrieve relevant chunks and generate a grounded answer with [Source N] citations."""
    hits = retrieve(query, k=k)

    if not hits:
        return {
            "query": query,
            "answer": "No relevant material found in the corpus. Run ingest_docs.py first.",
            "sources": [],
        }

    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[Source {i} | {hit['source']}, page {hit['page']}]\n{hit['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are an expert on the EU AI Act and related AI governance frameworks.\n\n"
        "Answer the user's question using ONLY the source material below. "
        "Cite sources using [Source N] notation. If the answer is not contained "
        "in the sources, state that explicitly rather than guessing.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"SOURCE MATERIAL:\n{context}\n\n"
        "ANSWER (with [Source N] citations):"
    )

    response = generate(prompt)

    return {
        "query": query,
        "answer": response,
        "sources": hits,
    }
