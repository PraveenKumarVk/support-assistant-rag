from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
BM25_INDEX_PATH = BASE_DIR / "bm25.pkl"
