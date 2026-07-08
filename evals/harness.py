import json
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

from src import pipeline
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import faithfulness, context_precision, context_recall, answer_correctness

FAITHFULNESS_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.60

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
RESULTS_MD = RESULTS_DIR / "results.md"

def normalize_source(path: str) -> str:
    return path.split("corpus/")[-1]

def run_queries(queries: list) -> list:
    results = []
    for item in queries:
        response, retrieved_chunks = pipeline.run_pipeline(item["query"])
        retrieved_sources = {normalize_source(c["file_path"]) for c in retrieved_chunks}
        expected_sources = {normalize_source(s) for s in item["expected_sources"]}
        results.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "category": item["category"],
            "question": item["query"],
            "contexts": [c["chunk_text"] for c in retrieved_chunks],
            "answer": response.answer,
            "ground_truth": item["ground_truth_answer"],
            "confidence": response.confidence,
            "sources_hit": bool(retrieved_sources & expected_sources),
        })
    return results

def run_ragas_eval(results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)
    llm = llm_factory("gpt-4o-mini", client=OpenAI(), max_tokens=8192)
    embeddings = OpenAIEmbeddings(client=AsyncOpenAI())
    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_correctness],
        llm=llm,
        embeddings=embeddings,
    )

def write_json(results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump({
            "results": results,
            "ragas_scores": scores,
            "thresholds": {
                "faithfulness": {"value": scores.get("faithfulness"), "min": FAITHFULNESS_THRESHOLD,
                                  "passed": scores.get("faithfulness", 0) >= FAITHFULNESS_THRESHOLD},
                "context_precision": {"value": scores.get("context_precision"), "min": CONTEXT_PRECISION_THRESHOLD,
                                       "passed": scores.get("context_precision", 0) >= CONTEXT_PRECISION_THRESHOLD},
            },
        }, f, indent=2)

def write_markdown(results: list, scores: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval Report", "", "## Ragas Scores", ""]
    for metric, value in scores.items():
        lines.append(f"- **{metric}**: {value:.3f}")

    faithfulness_score = scores.get("faithfulness", 0)
    context_precision_score = scores.get("context_precision", 0)
    lines += ["", "## Pass Thresholds", "",
              f"- Faithfulness ≥ {FAITHFULNESS_THRESHOLD}: {'PASS' if faithfulness_score >= FAITHFULNESS_THRESHOLD else 'FAIL'} ({faithfulness_score:.3f})",
              f"- Context Precision ≥ {CONTEXT_PRECISION_THRESHOLD}: {'PASS' if context_precision_score >= CONTEXT_PRECISION_THRESHOLD else 'FAIL'} ({context_precision_score:.3f})"]

    lines += ["", "## Per-Query Results", "", "| ID | Difficulty | Category | Sources Hit | Confidence |",
              "|----|-----------|----------|-------------|------------|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['difficulty']} | {r['category']} | {r['sources_hit']} | {r['confidence']} |")

    RESULTS_MD.write_text("\n".join(lines) + "\n")


def main():
    with open(Path(__file__).resolve().parent / "test_set.json", "r") as f:
        queries = json.load(f)["queries"]

    results = run_queries(queries)
    eval_result = run_ragas_eval(results)
    scores = eval_result.to_pandas().mean(numeric_only=True).to_dict()

    write_json(results, scores)
    write_markdown(results, scores)

    for result in results:
        print(f"[{result['id']}] sources_hit={result['sources_hit']} confidence={result['confidence']}")
    print(scores)


if __name__ == "__main__":
    main()
