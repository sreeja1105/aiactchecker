"""
Retrieval pipeline with optional source filtering.

Query -> embedding -> similarity search (filtered by source if specified) -> top-K chunks with citations.
"""

from typing import List, Dict, Union, Optional

from src.ingestion import get_chroma_collection
from src.llm import embed, generate


def retrieve(
    query: str,
    k: int = 5,
    source_filter: Optional[Union[str, List[str]]] = None,
) -> List[Dict]:
    """
    Retrieve top-K most relevant chunks for a query.

    Args:
        query: The search query.
        k: Number of chunks to retrieve.
        source_filter: Optional source name(s) to restrict the search.
            - None: search all sources
            - str: filter to a single source (e.g. "eu_ai_act")
            - list: filter to multiple sources (e.g. ["eu_ai_act", "gdpr"])
    """
    collection = get_chroma_collection()
    query_embedding = embed(query)

    where = None
    if source_filter:
        if isinstance(source_filter, str):
            where = {"source": source_filter}
        else:
            where = {"source": {"$in": list(source_filter)}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
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


def answer_with_citations(
    query: str,
    k: int = 5,
    source_filter: Optional[Union[str, List[str]]] = None,
) -> Dict:
    """
    Retrieve relevant chunks and generate a grounded answer with [Source N] citations.

    Args:
        query: The user's question.
        k: Number of chunks to retrieve.
        source_filter: Optional source(s) to restrict retrieval to specific framework(s).
    """
    hits = retrieve(query, k=k, source_filter=source_filter)

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
        "You are an expert on the EU AI Act, GDPR, and the NIST AI Risk Management Framework.\n\n"
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
