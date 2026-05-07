"""Local sentence-transformer embeddings for offline citation analysis.

For a single paper (~50-300 references) we just hold the vectors in a numpy
array and use a dot product — no index needed. FAISS only buys you something
once the corpus is large enough that exhaustive cosine becomes a real cost.

Empirically, on a typical CPU:
    - numpy dot of (N, 384) vs (384,) takes ~ N * 0.5 microseconds.
    - At N = 200, that's ~0.1 ms per query — instant.
    - At N = 10,000, ~5 ms per query — still fine for interactive use.
    - At N = 100,000+, exhaustive search starts to feel slow on every keystroke,
      and an HNSW / IVF FAISS index pays for itself.

Rule of thumb: numpy below ~10k entries; FAISS above. The `build_index` helper
returns a numpy-backed object with the same .search() shape as a FAISS index,
and switches to FAISS automatically past the threshold so calling code is
unchanged.

Nothing leaves the machine after the model is downloaded on first use.
"""

from functools import lru_cache
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_THRESHOLD = 10_000  # number of entries above which FAISS is worth pulling in


@lru_cache(maxsize=2)
def _load_model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for the embedding scorer. "
            "Install it with: pip install bib-checker[embeddings] "
            "(or use --scorer tfidf for a lighter alternative)"
        ) from e
    return SentenceTransformer(name)


def encode(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode strings to L2-normalized float32 vectors (so dot product = cosine)."""
    model = _load_model(model_name)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors."""
    return float(np.dot(vec_a, vec_b))


class _NumpyIndex:
    """Same .search() shape as a FAISS IndexFlatIP — but no FAISS dependency."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors  # already L2-normalized

    def search(self, query: np.ndarray, k: int):
        # query: (1, dim) or (Q, dim) — both are fine
        scores = query @ self.vectors.T  # (Q, N)
        k = min(k, scores.shape[1])
        # Top-k per query, descending
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        # Sort the top-k for stable output
        sorted_idx = np.take_along_axis(
            idx, np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1), axis=1
        )
        sorted_scores = np.take_along_axis(scores, sorted_idx, axis=1)
        return sorted_scores, sorted_idx


def build_index(entries: dict, model_name: str = DEFAULT_MODEL):
    """Build a similarity index over (abstract or title fallback) per entry.

    Returns (index, key_order, vectors). Index has a .search(query, k) method.
    Uses numpy for small corpora and FAISS once it crosses FAISS_THRESHOLD.
    """
    keys, texts = [], []
    for key, entry in entries.items():
        text = entry.get("abstract") or entry.get("title") or ""
        if not text.strip():
            continue
        keys.append(key)
        texts.append(text)

    if not keys:
        return None, [], None

    vectors = encode(texts, model_name=model_name)

    if len(keys) >= FAISS_THRESHOLD:
        try:
            import faiss
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return index, keys, vectors
        except ImportError:
            pass  # fall through to numpy

    return _NumpyIndex(vectors), keys, vectors
