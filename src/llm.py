"""
Hybrid LLM wrapper for AIActChecker.

Embeddings: sentence-transformers (local, no rate limits, no API costs).
Generation: Google Gemini with automatic retry on rate-limit errors.
"""

import re
import time
import random

from google import genai
from google.genai.errors import ClientError
from sentence_transformers import SentenceTransformer

from src.config import GEMINI_API_KEY, GEMINI_LLM_MODEL, LOCAL_EMBED_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

# Local embedding model. Loaded once at import.
_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)


def generate(prompt: str, model: str = GEMINI_LLM_MODEL, max_retries: int = 5) -> str:
    """
    Generate text from a prompt using Gemini LLM.
    Retries automatically on 429 rate-limit errors using Google's suggested delay.
    """
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(model=model, contents=prompt)
            return response.text
        except ClientError as e:
            err_str = str(e)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise

            wait_time = (2 ** attempt) + random.uniform(0, 1)
            match = re.search(r"retry in ([\d.]+)s", err_str)
            if match:
                wait_time = max(wait_time, float(match.group(1)) + 2)
            wait_time = min(wait_time, 90)

            print(f"[retry {attempt + 1}/{max_retries}] rate limited, waiting {wait_time:.1f}s")
            time.sleep(wait_time)

    raise RuntimeError(
        f"Generation failed after {max_retries} retries. Daily quota likely exhausted. "
        f"Try gemini-2.5-flash-lite (highest free tier) or wait until quota resets."
    )


def embed(text: str, task_type: str = None) -> list[float]:
    """Generate an embedding vector for a single text using the local model."""
    embedding = _embedder.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], task_type: str = None) -> list[list[float]]:
    """Batch-embed a list of texts using the local model."""
    embeddings = _embedder.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    return embeddings.tolist()
