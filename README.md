# Support Assistant RAG

A retrieval-augmented generation assistant over **Helix**, a fictional B2B SaaS workflow platform whose entire knowledge base lives in `corpus/` — 60 Markdown product docs, 25 PDF runbooks (5 of them scanned images), and 15 HTML support tickets. The pipeline ingests all three formats into a chunk store, embeds each chunk with `all-MiniLM-L6-v2` and pushes it to a Pinecone index while also building a local BM25 index over the same chunks, then answers a query by fusing dense and sparse retrieval (Reciprocal Rank Fusion) and passing the top-k chunks to `gpt-4o-mini` (via `ChatOpenAI`) with a context-only prompt that returns a structured `{answer, sources, confidence}` object. An eval harness built on `ragas` scores the pipeline against a 50-question test set for faithfulness, context precision/recall, and answer correctness.

## Architecture

```
                 ┌───────────────────────────────────────────────┐
                 │                    ingest.py                   │
                 │                                                 │
 corpus/         │  .md  ──► chunk_markdown_file ──┐               │
 ├─ product-docs/│  .html ─► chunk_html_file ───────┼─► chunk_store│
 ├─ runbooks/*.pdf  .pdf  ─► detect_pdf_type          (dict of      │
 └─ tickets/*.html         ├─ text-native ─► pypdf     chunk_id →  │
                           └─ scanned ─────► OCR       text+meta)  │
                 │                                    │            │
                 │                embed_chunks() ──────┤            │
                 │         (all-MiniLM-L6-v2, 384-dim)  │            │
                 │                                    ▼            │
                 │      ┌─────────────┐      ┌──────────────────┐  │
                 │      │ Pinecone    │      │ bm25.pkl         │  │
                 │      │ (dense,     │      │ (BM25Okapi over  │  │
                 │      │ cosine)     │      │ tokenized chunks)│  │
                 │      └─────────────┘      └──────────────────┘  │
                 └───────────────────────────────────────────────┘

                 ┌───────────────────────────────────────────────┐
                 │                  pipeline.py                   │
                 │                                                 │
  query ─────────┼──► retrieve.py                                  │
                 │      ├─ dense: Pinecone top (k*2)                │
                 │      ├─ sparse: BM25 top (k*2)                   │
                 │      └─ fuse via Reciprocal Rank Fusion ─► top-k │
                 │                        │                         │
                 │                        ▼                         │
                 │                  generate.py                     │
                 │      prompt(query, context) ──► gpt-4o-mini       │
                 │      ──► StructuredResponse{answer,sources,      │
                 │                              confidence}         │
                 └───────────────────────────────────────────────┘

                 ┌───────────────────────────────────────────────┐
                 │              evals/harness.py                  │
                 │  test_set.json (50 Q/A) ──► run_pipeline() for  │
                 │  each query ──► ragas.evaluate(faithfulness,    │
                 │  context_precision, context_recall,             │
                 │  answer_correctness) ──► results.json/.md       │
                 └───────────────────────────────────────────────┘
```

## Chunking strategy

All formats are chunked with a fixed-size sliding window: **1000 characters, 200-character overlap** (`chunk_text` in `src/ingest.py`), but each format is pre-processed differently before that window is applied:

- **Markdown** — the raw file is read as-is and chunked directly. Headings and code fences stay embedded in the surrounding text rather than being stripped, since Markdown structure is already fairly chunk-friendly and the docs are short enough that this rarely splits a table or code block awkwardly.
- **HTML (tickets)** — parsed with BeautifulSoup; script/style/nav/header/footer tags are stripped, then the document is split into **sections by heading** (`h1`–`h6`) before the 1000/200 window is applied *within* each section. Tickets are long, multi-turn conversations, so chunking by heading first keeps a given exchange together instead of drawing arbitrary cuts across unrelated turns; the resolution (which the corpus notes usually lands in the final agent message) is far more likely to survive intact in one chunk.
- **PDF (runbooks)** — `detect_pdf_type` first checks whether `pypdf` can extract a meaningful amount of text per page (>70 chars/page average). Text-native PDFs are extracted directly with `pypdf`. Scanned (image-only) PDFs are rendered to images with `pdf2image` (300 DPI) and OCR'd with `pytesseract`, rather than being skipped — the corpus explicitly includes 5 scanned runbooks and several eval queries (e.g. Q047, Q048) depend on their content, so OCR was the only way to make that content retrievable at all. The OCR'd text is then chunked the same way as any other plain text.

Every chunk carries `file_path`, `file_type`, and `location` (filename, or heading text for HTML) in its metadata, which is what lets retrieval report *which* source document an answer came from and lets the eval harness check that against `expected_sources`.

## Retrieval strategy

**Hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion (RRF)** (`src/retrieve.py`):

1. Embed the query with the same `all-MiniLM-L6-v2` model used at ingest time and query Pinecone for the top `k*2` matches (dense/semantic).
2. Tokenize the query and score it against the same corpus with `BM25Okapi`, taking the top `k*2` matches (sparse/lexical).
3. Fuse both ranked lists with RRF (`score = Σ 1/(60 + rank)` per list) and take the final top-`k` by fused score.

I chose hybrid retrieval over reranking or multi-query expansion because the corpus mixes free-form prose (docs, tickets) with a lot of **exact tokens that matter** — status codes (`429`), header names (`X-Helix-Signature`), dollar amounts (`$299`), plan names (`Enterprise`). Dense embeddings are good at semantic proximity but can under-rank a chunk that is a near-verbatim keyword match, which is exactly what BM25 is strong at; RRF combines the two without needing to tune a blend weight.

**Before vs. after**, measured by actually disabling BM25 fusion and re-running the same 50-query test set through `evals/harness.py` (dense-only: Pinecone top-`k` only; hybrid: the current `retrieve.py`, dense + BM25 fused with RRF):

| Metric | Dense-only | Hybrid (dense + BM25 + RRF) |
|---|---|---|
| Faithfulness | 0.964 | **0.966** |
| Context precision | 0.840 | **0.862** |
| Context recall | 0.894 | **0.908** |
| Answer correctness | **0.563** | 0.557 |
| `sources_hit` (raw, 50 queries) | **48/50** | 47/50 |

The honest result: on this particular 50-query set, hybrid's edge is real but modest — it improves context precision and recall by 1–2 points, which is the metric it's actually meant to move (surfacing more of the right chunks, fewer wrong ones), but it doesn't move faithfulness or answer correctness meaningfully, and it's not strictly better query-by-query. One query (Q003, see below) that dense-only alone answers correctly gets *worse* under fusion — RRF is not monotonic per-query, since a chunk ranked #2 by dense but absent from BM25's list can get pushed below chunks both retrievers rank moderately. I kept hybrid because the aggregate context precision/recall gain plus the qualitative robustness against exact-token queries (rate limit codes, header names) outweigh the occasional single-query regression, but it's a real tradeoff, not a strict win.

## Generation prompt

```text
Act as a support assistant and answer the query given the context. Your answer should be solely based on provided evidence.
You cannot use external knowledge and any information other than the context provided to answer the query.
Query: {query},
Context:{chunk_text}
```

(`src/generate.py`, via `ChatPromptTemplate` → `ChatOpenAI` → Pydantic-parsed `StructuredResponse{answer, sources, confidence}`.)

Reasoning behind the structure:

- **Role framing ("Act as a support assistant")** sets tone and scope expectations before the task instruction, since this is meant to sit behind a support tool, not answer arbitrary open-domain questions.
- **Explicit "no external knowledge" constraint, stated twice** (once as a positive instruction, once as a negative one) is the main lever against hallucination given this is a fictional company — the model's own training data about "Helix" or generic SaaS conventions is actively wrong here, so the prompt has to close that door hard rather than rely on the model to infer it.
- **Query before context** keeps the model anchored on what it needs to answer before it reads the (often long, multi-chunk) evidence block, which is inserted as one flat block rather than as source-tagged snippets — a simplification `sources` currently papers over by being populated separately from the retrieved chunks' `file_path` metadata rather than parsed out of the model's own citations.
- **Confidence is not model-generated** — it's derived after the LLM call from the top dense similarity score (`>0.85` high, `>0.7` medium, else low). This keeps the confidence signal grounded in retrieval quality rather than the model's own (often overconfident) self-assessment, though it currently only looks at the dense score and ignores the fused/BM25 signal.

## Eval results

Full per-query results: `evals/results/results.json` / `evals/results/results.md`. Ragas scores over the 50-query test set:

| Metric | Score | Threshold | Result |
|---|---|---|---|
| Faithfulness | 0.966 | ≥ 0.70 | PASS |
| Context precision | 0.862 | ≥ 0.60 | PASS |
| Context recall | 0.908 | — | — |
| Answer correctness | 0.557 | — | — |

### Representative failure cases

**Q003 — "How much does the Pro plan cost?"**
`sources_hit: False` under hybrid, but `True` under the dense-only ablation — dense-only actually retrieved `product-docs/billing/plans.md` at rank 2 and answered "$299/month" correctly. Under hybrid, that same chunk got fused out of the final top-5: BM25's ranked list for this short, generic query ("Pro plan cost") surfaced several other billing-tagged chunks (a downgrade runbook, a plan-availability snippet, an overage ticket) that all share the word "plan," and RRF's rank-based scoring let those crowd out a chunk that dense ranked highly but BM25 didn't rank at all. The final hybrid answer ("$299", missing "/month") still happened to be directionally correct because the dollar figure leaked in from a downgrade-scenario chunk, but it's a clear example of RRF being non-monotonic per query — fusion doesn't strictly dominate either retriever it's built from.

**Q002 — "How long can a workflow run before it times out?"**
`sources_hit: False` despite a fully correct, well-supported answer. Retrieval pulled the 30-minute limit from a support ticket and an incident summary rather than from `product-docs/workflows/creating-workflows.md`, the one path listed in `expected_sources`. This is the strict `sources_hit` metric being stricter than the actual task — the corpus deliberately repeats the same fact (30-minute timeout) across multiple documents, and retrieval correctly found *a* correct source, just not *the* one the test set pinned. It's a scoring-methodology gap more than a pipeline defect, but it shows `sources_hit` alone isn't sufficient as a precision signal without also checking answer correctness.

**Q018 — "What integrations does Helix support?"**
Answer was actually wrong (talked about mobile-app integration *management* instead of listing supported integrations), because retrieval surfaced `mobile-limitations.md` and mobile-app docs instead of `product-docs/integrations/overview.md`. The query's dominant term "integrations" apparently overlapped more with mobile docs that mention "integration management" in passing than with the actual integrations overview doc — a case where BM25's term overlap and the dense embedding both landed on the wrong side, and nothing in the fusion step corrected for it. This is the clearest true retrieval failure in the set and points at reranking as the next fix (see below).

## How to run

```bash
poetry install
cp .env.example .env   # fill in PINECONE_API_KEY and OPENAI_API_KEY
poetry run python -m src.ingest
poetry run python -c "from src.pipeline import run_pipeline; r,_=run_pipeline('What is the Pro plan rate limit?'); print(r)"
poetry run python -m evals.harness
```

## What I'd do with another week

1. **Add a reranking stage** — the clearest failure mode (Q018) is a wrong chunk winning both dense and sparse retrieval; a cross-encoder reranker over the fused top-k would catch mismatches RRF can't, since RRF only reorders what dense/BM25 already agree is plausible.
2. **Switch to token-based, structure-aware chunking** — chunking is currently character-based and format-agnostic (aside from HTML's heading split), which risks splitting mid-sentence, mid-table, or mid-code-block. Token-aware, structure-preserving chunking (e.g. never split inside a Markdown table or fenced code block) would likely lift context precision further.
3. **Fold BM25 score into the confidence heuristic** — confidence is currently derived only from the dense similarity score; incorporating the fused RRF score (or the BM25 score) would make confidence a better proxy for "was this actually a strong hybrid match," particularly for the exact-token queries hybrid retrieval was built to help with.
4. **Cite sources per-claim, not per-answer** — `sources` is populated from every retrieved chunk's `file_path` regardless of whether the model actually used it, so a low-relevance chunk still shows up as a cited source. Having the model tag which chunk(s) it drew from (or computing attribution post-hoc) would make `sources` trustworthy enough to show to an end user.
5. **Investigate the answer-correctness gap (0.557)** — faithfulness and context precision/recall are strong, but answer correctness lags well behind, meaning answers are often faithful to retrieved context and well-sourced but still miss specific facts the ground truth expects (see Q003). Worth a systematic pass through the medium/hard queries to see whether this is a chunk-size problem (facts split across chunk boundaries) or a prompt problem (model summarizing instead of stating the precise figure).
