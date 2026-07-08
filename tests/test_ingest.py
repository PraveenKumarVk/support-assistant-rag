import pytest

from src import ingest


def test_chunk_text_basic():
    chunks = ingest.chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_chunk_text_overlap_too_large_raises():
    with pytest.raises(ValueError):
        ingest.chunk_text("abcdef", chunk_size=3, overlap=3)


def test_load_markdown_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.load_markdown_file(str(tmp_path / "missing.md"))


def test_load_markdown_file_reads_content(tmp_path):
    file_path = tmp_path / "doc.md"
    file_path.write_text("# hello world")
    assert ingest.load_markdown_file(str(file_path)) == "# hello world"


def test_load_html_file_heading_sections(tmp_path):
    html = """
    <html><body>
        <h1>Intro</h1>
        <p>First section text.</p>
        <h2>Details</h2>
        <p>Second section text.</p>
    </body></html>
    """
    file_path = tmp_path / "doc.html"
    file_path.write_text(html)
    sections = ingest.load_html_file(str(file_path))
    headings = [s[0] for s in sections]
    assert "Intro" in headings
    assert "Details" in headings


def test_load_html_file_strips_script_and_nav(tmp_path):
    html = """
    <html><body>
        <nav><h3>Nav Link</h3></nav>
        <script>var x = 1;</script>
        <h1>Real Heading</h1>
        <p>Real content.</p>
    </body></html>
    """
    file_path = tmp_path / "doc.html"
    file_path.write_text(html)
    sections = ingest.load_html_file(str(file_path))
    headings = [s[0] for s in sections]
    assert headings == ["Real Heading"]
    assert "var x" not in sections[0][1]


def test_load_html_file_headingless_fallback(tmp_path):
    html = "<html><body><div><p>Just a paragraph, no headings.</p></div></body></html>"
    file_path = tmp_path / "doc.html"
    file_path.write_text(html)
    sections = ingest.load_html_file(str(file_path))
    assert len(sections) == 1
    assert sections[0][0] == ""
    assert "Just a paragraph" in sections[0][1]


def test_load_html_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.load_html_file(str(tmp_path / "missing.html"))


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_detect_pdf_type_text_native(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-fake")
    pages = [_FakePage("x" * 100) for _ in range(2)]
    monkeypatch.setattr(ingest.pypdf, "PdfReader", lambda f: _FakeReader(pages))
    assert ingest.detect_pdf_type(str(file_path)) == "text_native"


def test_detect_pdf_type_scanned(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-fake")
    pages = [_FakePage(None), _FakePage("")]
    monkeypatch.setattr(ingest.pypdf, "PdfReader", lambda f: _FakeReader(pages))
    assert ingest.detect_pdf_type(str(file_path)) == "scanned"


def test_load_text_native_pdf_handles_none_text(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-fake")
    pages = [_FakePage("first"), _FakePage(None)]
    monkeypatch.setattr(ingest.pypdf, "PdfReader", lambda f: _FakeReader(pages))
    assert ingest.load_text_native_pdf(str(file_path)) == "first\n"


def test_build_bm25_index_writes_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.config, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")
    chunk_store = {
        "id1": {"chunk_text": "workflow steps limit"},
        "id2": {"chunk_text": "billing invoice details"},
    }
    bm25, chunk_ids = ingest.build_bm25_index(chunk_store)
    assert chunk_ids == ["id1", "id2"]
    assert (tmp_path / "bm25.pkl").exists()


class _FakeIndex:
    def __init__(self):
        self.upserted = None

    def upsert(self, vectors, namespace):
        self.upserted = (vectors, namespace)


class _FakePineconeClient:
    def __init__(self):
        self.index = _FakeIndex()

    def has_index(self, name):
        return True

    def Index(self, name):
        return self.index


def test_store_chunks_uses_dict_key_as_vector_id(monkeypatch):
    fake_pc = _FakePineconeClient()
    monkeypatch.setattr(ingest, "pc", fake_pc)
    chunk_store = {
        "chunk-abc": {"chunk_text": "hello", "file_path": "a.md", "embedding": [0.1, 0.2]},
    }
    ingest.store_chunks(chunk_store)
    vectors, namespace = fake_pc.index.upserted
    assert namespace == "md-vectors"
    assert vectors[0]["id"] == "chunk-abc"
    assert vectors[0]["values"] == [0.1, 0.2]
    assert "embedding" not in vectors[0]["metadata"]
