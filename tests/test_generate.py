from src import generate


class _FakeChain:
    def __init__(self, content):
        self._content = content

    def invoke(self, inputs):
        return type("Response", (), {"content": self._content})()


class _FakePrompt:
    def __or__(self, llm):
        return _FakeChain("The workflow limit is 100 steps.")


def _patch_llm(monkeypatch):
    monkeypatch.setattr(generate.ChatPromptTemplate, "from_template", lambda template: _FakePrompt())
    monkeypatch.setattr(generate, "ChatOpenAI", lambda api_key: object())


def test_generate_returns_structured_response(monkeypatch):
    _patch_llm(monkeypatch)
    retrieved_chunks = [
        {"chunk_text": "chunk one", "file_path": "a.md"},
        {"chunk_text": "chunk two", "file_path": "b.md"},
    ]
    result = generate.generate("What's the step limit?", retrieved_chunks, top_similarity_score=0.9)

    assert result.answer == "The workflow limit is 100 steps."
    assert result.sources == ["a.md", "b.md"]
    assert result.confidence == "high"


def test_generate_confidence_medium(monkeypatch):
    _patch_llm(monkeypatch)
    result = generate.generate("q", [{"chunk_text": "c", "file_path": "a.md"}], top_similarity_score=0.75)
    assert result.confidence == "medium"


def test_generate_confidence_low(monkeypatch):
    _patch_llm(monkeypatch)
    result = generate.generate("q", [{"chunk_text": "c", "file_path": "a.md"}], top_similarity_score=0.2)
    assert result.confidence == "low"
