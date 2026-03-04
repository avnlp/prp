# PRP Documentation

- **[Ranker API](./rankers.md)** - Complete API documentation for `PairwiseRankingPrompting`
- **[Evaluation Metrics](./evaluation.md)** - Evaluator API, metrics definitions (NDCG, MAP, Recall, Precision), and usage examples
- **[Data Loading](./dataloader.md)** - Dataloader class, supported datasets, and dataset structure
- **[Configuration Management](./configuration.md)** - Configuration schemas, YAML examples, and loading patterns

## Core Concepts

### Pairwise Ranking

Unlike pointwise (scoring each document independently) or listwise (ranking all documents at once) approaches, PRP reduces the ranking task to simple pairwise comparisons. Given a query, the LLM is asked: "Which of these two passages is more relevant?"

This simplification is key—LLMs are much better at comparing two items than producing calibrated scores or full permutations. The pairwise preferences are then aggregated into a final ranking using efficient sorting algorithms.

### Ranker Interface

The ranker accepts a query and list of documents, returning documents sorted by relevance:

```python
reranker = PairwiseRankingPrompting(model_name="gpt-4o-mini")
ranked_docs = reranker.rerank(query, documents, method="heapsort")
```

Output is always a list of document strings ordered by relevance (most relevant first).

### Position Bias Mitigation

LLMs exhibit position bias—they may favor the first or second document regardless of relevance. PRP mitigates this by comparing each pair **bidirectionally**:

1. Compare A vs B → get preference
2. Compare B vs A → get preference
3. If both agree → clear winner
4. If they conflict → tie (0.5 points each)

This technique significantly improves ranking stability and quality.

### Ranking Methods

Three algorithms with different cost/quality trade-offs:

| Method | Complexity | Best For |
|--------|------------|----------|
| `allpairs` | O(N²) | Highest quality, small document sets |
| `heapsort` | O(N log N) | General purpose (recommended) |
| `sliding_k` | O(K×N) | Large sets, top-k optimization |

See [Ranker API](./rankers.md) for detailed algorithm descriptions.

### Evaluation Workflow

Evaluate ranker quality using standard IR metrics:

1. Load a dataset (FIQA, SciFact, NFCorpus, TREC-19/20)
2. Run ranker on queries
3. Compare results to ground-truth relevance judgments
4. Compute metrics (NDCG, MAP, Recall, Precision)

See [Evaluation Metrics](./evaluation.md) for detailed metric definitions.

## Common Patterns

### Basic Reranking

```python
from prp import PairwiseRankingPrompting

# Initialize ranker (works with any OpenAI-compatible API)
reranker = PairwiseRankingPrompting(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_key="your-api-key",
    base_url="https://api.groq.com/openai/v1"
)

# Rerank documents
query = "What causes climate change?"
documents = ["Document 1 text...", "Document 2 text...", "Document 3 text..."]

ranked = reranker.rerank(query, documents, method="heapsort")
```

### Method Selection

```python
# For best quality (small document sets)
ranked = reranker.rerank(query, documents, method="allpairs")

# For balanced cost/quality (recommended default)
ranked = reranker.rerank(query, documents, method="heapsort")

# For efficiency (large sets, only need top-k)
ranked = reranker.rerank(query, documents, method="sliding_k", top_k=10)
```

### Evaluation Pipeline

```python
from prp import PairwiseRankingPrompting
from prp.dataloader import Dataloader
from prp.evaluation import Evaluator, EvaluatorParams

# Load dataset
loader = Dataloader("beir/scifact/test")
dataset = loader.load()

# Run ranker and collect results
# ... (rerank each query's candidates)

# Evaluate
evaluator = Evaluator(
    relevance_judgments=dataset.relevance_judgments,
    run_results=run_results,
    config=EvaluatorParams(cutoff_values=(1, 5, 10))
)

metrics = evaluator.evaluate()
print(f"NDCG@10: {metrics.ndcg['NDCG@10']}")
```

### Token Tracking

```python
# Monitor API usage across multiple reranking calls
reranker = PairwiseRankingPrompting(model_name="gpt-4o-mini")

for query, docs in queries_and_docs:
    reranker.rerank(query, docs)

print(f"Total tokens used: {reranker.get_token_count()}")
```
