import pickle

from . import config
from .retrieve import retrieve
from .generate import generate

_bm25_cache = None


def _load_bm25():
    global _bm25_cache
    if _bm25_cache is None:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _bm25_cache = pickle.load(f)
    return _bm25_cache


def run_pipeline(query: str, k: int = 5):
    bm25, chunk_ids = _load_bm25()
    top_similarity_score, retrieved_chunks = retrieve(query, bm25, chunk_ids, k)
    response = generate(query, retrieved_chunks, top_similarity_score)
    return response, retrieved_chunks
