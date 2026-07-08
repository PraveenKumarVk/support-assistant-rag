from rank_bm25 import BM25Okapi

from src import retrieve


class _Match:
    def __init__(self, id, score, metadata):
        self.id = id
        self.score = score
        self.metadata = metadata


class _FakeQueryResult:
    def __init__(self, matches):
        self.matches = matches


class _FakeVector:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeFetchResult:
    def __init__(self, vectors):
        self.vectors = vectors


class _FakeIndex:
    def __init__(self, matches, fetchable_metadata=None):
        self._matches = matches
        self._fetchable_metadata = fetchable_metadata or {}
        self.fetch_calls = []

    def query(self, vector, top_k, include_metadata, namespace):
        return _FakeQueryResult(self._matches[:top_k])

    def fetch(self, ids, namespace):
        self.fetch_calls.append(ids)
        return _FakeFetchResult({id: _FakeVector(self._fetchable_metadata[id]) for id in ids if id in self._fetchable_metadata})


def _build_bm25(documents):
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def test_retrieve_combines_dense_and_sparse_results(monkeypatch):
    chunk_ids = ["c1", "c2", "c3", "c4"]
    documents = [
        "workflow step limit is one hundred",
        "billing invoice cycle details",
        "unrelated api rate limiting notes",
        "mobile app notification settings",
    ]
    bm25 = _build_bm25(documents)

    dense_matches = [
        _Match("c2", 0.9, {"chunk_text": documents[1], "file_path": "b.md"}),
        _Match("c1", 0.8, {"chunk_text": documents[0], "file_path": "a.md"}),
    ]
    fake_index = _FakeIndex(dense_matches)

    monkeypatch.setattr(retrieve, "get_index", lambda: fake_index)
    monkeypatch.setattr(retrieve.embed_model, "encode", lambda query: type("V", (), {"tolist": lambda self: [0.0]})())

    top_score, retrieved_chunks = retrieve.retrieve("workflow step limit", bm25, chunk_ids, k=2)

    assert top_score == 0.9
    assert len(retrieved_chunks) <= 2
    assert all("chunk_text" in c for c in retrieved_chunks)


def test_retrieve_fetches_metadata_for_sparse_only_hits(monkeypatch):
    chunk_ids = ["c1", "c2", "c3", "c4"]
    documents = [
        "unrelated content about billing",
        "unrelated content about mobile",
        "workflow step limit is one hundred exactly",
        "another unrelated document",
    ]
    bm25 = _build_bm25(documents)

    # Dense search returns only "c1"; "c3" is the clear BM25 winner but never appears
    # in the dense results, so its metadata must come from an explicit fetch() call.
    dense_matches = [
        _Match("c1", 0.5, {"chunk_text": documents[0], "file_path": "a.md"}),
    ]
    fake_index = _FakeIndex(dense_matches, fetchable_metadata={
        "c3": {"chunk_text": documents[2], "file_path": "c.md"},
    })

    monkeypatch.setattr(retrieve, "get_index", lambda: fake_index)
    monkeypatch.setattr(retrieve.embed_model, "encode", lambda query: type("V", (), {"tolist": lambda self: [0.0]})())

    top_score, retrieved_chunks = retrieve.retrieve("workflow step limit one hundred", bm25, chunk_ids, k=2)

    retrieved_files = {c["file_path"] for c in retrieved_chunks}
    assert "c.md" in retrieved_files
    assert fake_index.fetch_calls


def test_retrieve_returns_zero_score_when_no_matches(monkeypatch):
    documents = ["doc one text", "doc two text", "doc three text", "doc four text"]
    chunk_ids = ["c1", "c2", "c3", "c4"]
    bm25 = _build_bm25(documents)
    fake_index = _FakeIndex([])

    monkeypatch.setattr(retrieve, "get_index", lambda: fake_index)
    monkeypatch.setattr(retrieve.embed_model, "encode", lambda query: type("V", (), {"tolist": lambda self: [0.0]})())

    top_score, retrieved_chunks = retrieve.retrieve("anything", bm25, chunk_ids, k=1)

    assert top_score == 0
