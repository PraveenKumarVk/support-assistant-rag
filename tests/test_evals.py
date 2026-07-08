import json

from evals import harness


def test_normalize_source_strips_absolute_prefix():
    assert harness.normalize_source("/Users/me/repo/corpus/product-docs/a.md") == "product-docs/a.md"


def test_normalize_source_passthrough_for_relative_path():
    assert harness.normalize_source("product-docs/a.md") == "product-docs/a.md"


class _FakeResponse:
    def __init__(self, answer, sources, confidence):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence


def test_run_queries_collects_expected_fields(monkeypatch):
    fake_response = _FakeResponse("100 steps", ["/repo/corpus/product-docs/workflows.md"], "high")
    fake_chunks = [{"chunk_text": "the limit is 100 steps", "file_path": "/repo/corpus/product-docs/workflows.md"}]

    monkeypatch.setattr(harness.pipeline, "run_pipeline", lambda query: (fake_response, fake_chunks))

    queries = [{
        "id": "Q001",
        "query": "What's the step limit?",
        "ground_truth_answer": "100 steps",
        "expected_sources": ["product-docs/workflows.md"],
        "difficulty": "easy",
        "category": "workflows",
    }]

    results = harness.run_queries(queries)

    assert len(results) == 1
    result = results[0]
    assert result["question"] == "What's the step limit?"
    assert result["answer"] == "100 steps"
    assert result["ground_truth"] == "100 steps"
    assert result["contexts"] == ["the limit is 100 steps"]
    assert result["sources_hit"] is True


def test_run_queries_sources_miss(monkeypatch):
    fake_response = _FakeResponse("answer", ["/repo/corpus/tickets/unrelated.html"], "low")
    fake_chunks = [{"chunk_text": "unrelated text", "file_path": "/repo/corpus/tickets/unrelated.html"}]

    monkeypatch.setattr(harness.pipeline, "run_pipeline", lambda query: (fake_response, fake_chunks))

    queries = [{
        "id": "Q002",
        "query": "irrelevant query",
        "ground_truth_answer": "gt",
        "expected_sources": ["product-docs/workflows.md"],
        "difficulty": "easy",
        "category": "workflows",
    }]

    results = harness.run_queries(queries)
    assert results[0]["sources_hit"] is False


def test_write_json_and_markdown(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(harness, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(harness, "RESULTS_JSON", results_dir / "results.json")
    monkeypatch.setattr(harness, "RESULTS_MD", results_dir / "results.md")

    results = [{"id": "Q001", "difficulty": "easy", "category": "workflows", "sources_hit": True, "confidence": "high"}]
    scores = {"faithfulness": 0.8, "context_precision": 0.65, "context_recall": 0.7, "answer_correctness": 0.75}

    harness.write_json(results, scores)
    harness.write_markdown(results, scores)

    assert (results_dir / "results.json").exists()
    assert (results_dir / "results.md").exists()

    data = json.loads((results_dir / "results.json").read_text())
    assert data["thresholds"]["faithfulness"]["passed"] is True
    assert data["thresholds"]["context_precision"]["passed"] is True

    markdown = (results_dir / "results.md").read_text()
    assert "Q001" in markdown
    assert "PASS" in markdown


def test_write_json_reports_threshold_failure(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(harness, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(harness, "RESULTS_JSON", results_dir / "results.json")
    monkeypatch.setattr(harness, "RESULTS_MD", results_dir / "results.md")

    scores = {"faithfulness": 0.5, "context_precision": 0.4, "context_recall": 0.6, "answer_correctness": 0.6}
    harness.write_json([], scores)

    data = json.loads((results_dir / "results.json").read_text())
    assert data["thresholds"]["faithfulness"]["passed"] is False
    assert data["thresholds"]["context_precision"]["passed"] is False
