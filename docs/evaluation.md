# Evaluation API

Documentation for the Evaluator class and evaluation metrics.

## Evaluator

Information Retrieval evaluation engine using pytrec_eval.

### Constructor

```python
Evaluator(
    relevance_judgments: dict[str, dict[str, int]],
    run_results: dict[str, dict[str, float]],
    config: EvaluatorParams | None = None,
)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `relevance_judgments` | `dict[str, dict[str, int]]` | Yes | Ground truth relevance judgments. Maps query_id to {doc_id: relevance_score} |
| `run_results` | `dict[str, dict[str, float]]` | Yes | System run results. Maps query_id to {doc_id: score} |
| `config` | `EvaluatorParams \| None` | No | Configuration for evaluation. Uses defaults if not provided |

### Methods

#### `evaluate()`

Compute evaluation metrics and return results.

```python
evaluate() -> EvaluationMetrics
```

##### Returns

`EvaluationMetrics`: Immutable container with averaged evaluation metrics.

#### `evaluation_metrics` (property)

Access computed metrics after calling `evaluate()`.

```python
@property
evaluation_metrics -> EvaluationMetrics
```

##### Raises

`RuntimeError`: If `evaluate()` has not been called yet.

## EvaluatorParams

Configuration dataclass for evaluation parameters.

```python
@dataclass
class EvaluatorParams:
    cutoff_values: tuple[int, ...] = (1, 3, 5, 10)
    ignore_identical_ids: bool = True
    decimal_precision: int = 4
    metrics_to_compute: tuple[str, ...] = ("ndcg", "map", "recall", "precision")
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cutoff_values` | `tuple[int, ...]` | `(1, 3, 5, 10)` | Cutoff levels for metrics (e.g., NDCG@1, NDCG@10) |
| `ignore_identical_ids` | `bool` | `True` | Exclude docs where query_id == doc_id (prevents leakage) |
| `decimal_precision` | `int` | `4` | Decimal places for rounding metric values |
| `metrics_to_compute` | `tuple[str, ...]` | `("ndcg", "map", "recall", "precision")` | Which metrics to compute |

## EvaluationMetrics

Immutable container for evaluation results.

```python
@dataclass(frozen=True)
class EvaluationMetrics:
    ndcg: dict[str, float]
    map: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `ndcg` | `dict[str, float]` | NDCG scores keyed by metric string (e.g., `"NDCG@10": 0.7234`) |
| `map` | `dict[str, float]` | MAP scores keyed by metric string |
| `recall` | `dict[str, float]` | Recall scores keyed by metric string |
| `precision` | `dict[str, float]` | Precision scores keyed by metric string |

## Metrics Explained

### NDCG (Normalized Discounted Cumulative Gain)

Measures ranking quality by discounting relevance scores based on position. Higher positions contribute more to the score.

- **Range**: 0.0 to 1.0
- **Best for**: Graded relevance judgments
- **Use case**: When document position matters (e.g., search results)

### MAP (Mean Average Precision)

Balances precision and recall for binary relevance scenarios. Averages precision at each relevant document.

- **Range**: 0.0 to 1.0
- **Best for**: Binary relevance (relevant/not relevant)
- **Use case**: When you need a single summary metric

### Recall@k

Measures coverage - the fraction of relevant documents retrieved in top-k results.

- **Range**: 0.0 to 1.0
- **Formula**: `(relevant docs in top-k) / (total relevant docs)`
- **Use case**: When finding all relevant documents matters

### Precision@k

Measures result quality - the fraction of retrieved documents that are relevant.

- **Range**: 0.0 to 1.0
- **Formula**: `(relevant docs in top-k) / k`
- **Use case**: When result quality matters more than coverage

## Usage Examples

### Basic Evaluation

```python
from prp.evaluation import Evaluator, EvaluatorParams

# Ground truth relevance judgments
qrels = {
    "q1": {"doc1": 2, "doc2": 1, "doc3": 0},
    "q2": {"doc4": 1, "doc5": 2},
}

# System results (higher score = more relevant)
results = {
    "q1": {"doc1": 0.9, "doc2": 0.7, "doc3": 0.5},
    "q2": {"doc5": 0.8, "doc4": 0.6},
}

# Create evaluator and compute metrics
evaluator = Evaluator(qrels, results)
metrics = evaluator.evaluate()

print(metrics.ndcg)  # {"NDCG@1": 0.8, "NDCG@3": 0.75, ...}
print(metrics.map)   # {"MAP@1": 0.7, "MAP@3": 0.72, ...}
```

### Custom Configuration

```python
# Only compute NDCG and MAP at cutoffs 5 and 10
params = EvaluatorParams(
    cutoff_values=(5, 10),
    metrics_to_compute=("ndcg", "map"),
    decimal_precision=3,
)

evaluator = Evaluator(qrels, results, config=params)
metrics = evaluator.evaluate()
```

### Complete Workflow with Ranker

```python
from prp import PairwiseRankingPrompting
from prp.evaluation import Evaluator
from prp.dataloader import Dataloader

# Load dataset
loader = Dataloader("beir/scifact/test")
dataset = loader.load()

# Initialize ranker
ranker = PairwiseRankingPrompting(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-api-key"
)

# Rerank documents for each query
results = {}
for query_id, query_text in dataset.queries.items():
    # Get candidate documents (e.g., from BM25)
    candidates = get_candidates(query_id)  # Your retrieval function

    # Rerank with PRP
    ranked = ranker.rerank(query_text, candidates, method="heapsort")

    # Store results with scores
    results[query_id] = {doc_id: 1.0 / (i + 1) for i, doc_id in enumerate(ranked)}

# Evaluate
evaluator = Evaluator(dataset.relevance_judgments, results)
metrics = evaluator.evaluate()

print(f"NDCG@10: {metrics.ndcg['NDCG@10']}")
print(f"MAP@10: {metrics.map['MAP@10']}")
```

## Metric Selection Guide

| Use Case | Recommended Metrics |
|----------|-------------------|
| General ranking quality | NDCG@10 |
| Binary relevance | MAP, Precision |
| Finding all relevant docs | Recall |
| Top-heavy ranking | NDCG@1, NDCG@3 |
| Comprehensive evaluation | All metrics at multiple cutoffs |
