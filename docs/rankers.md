# Ranker API

Complete API documentation for `PairwiseRankingPrompting`.

## PairwiseRankingPrompting

The main class for pairwise document reranking with position bias mitigation.

### Constructor

```python
PairwiseRankingPrompting(
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
    client_kwargs: dict[str, Any] | None = None,
    completion_kwargs: dict[str, Any] | None = None,
)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `str` | Yes | LLM model identifier (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`, `"gpt-4o-mini"`) |
| `api_key` | `str \| None` | No | API key for authentication. Defaults to `OPENAI_API_KEY` environment variable |
| `base_url` | `str \| None` | No | Custom API endpoint URL for non-OpenAI providers |
| `client_kwargs` | `dict` | No | Additional kwargs passed to the OpenAI client (e.g., timeouts, proxies) |
| `completion_kwargs` | `dict` | No | Default parameters for completion calls (e.g., `temperature`, `max_tokens`) |

### Methods

#### `rerank()`

Rerank a list of documents by relevance to a query.

```python
rerank(
    query: str,
    documents: list[str],
    method: str = "allpairs",
    top_k: int | None = None,
    sliding_k_passes: int | None = None,
) -> list[str]
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | - | User search query string |
| `documents` | `list[str]` | - | List of document texts to be reranked |
| `method` | `str` | `"allpairs"` | Ranking algorithm: `"allpairs"`, `"heapsort"`, or `"sliding_k"` |
| `top_k` | `int \| None` | `None` | Limits results to top k documents. If `None`, returns all documents |
| `sliding_k_passes` | `int \| None` | `None` | Number of passes for `sliding_k` method (defaults to `top_k` or 10) |

##### Returns

`list[str]`: Documents sorted by descending relevance.

##### Raises

`ValueError`: If an unsupported method is provided.

#### `get_token_count()`

Retrieve the cumulative number of tokens used across all API calls.

```python
get_token_count() -> int
```

##### Returns

`int`: Total tokens consumed by this ranker instance.

## Ranking Methods

### `allpairs` (O(N²))

Enumerates all document pairs and performs global score aggregation.

- **Scoring**: Winner gets 1.0 point, tie gives 0.5 points each
- **Parallelizable**: Yes (all comparisons are independent)
- **Order Sensitivity**: Low
- **Best for**: Highest quality results on smaller document sets

### `heapsort` (O(N log N))

Uses LLM pairwise preferences as a comparator in HeapSort algorithm.

- **Parallelizable**: No (sequential sorting operations)
- **Order Sensitivity**: Low
- **Best for**: General purpose use (recommended default)

### `sliding_k` (O(K×N))

Bubble-sort-inspired sliding window that optimizes for top-k results.

- **Parallelizable**: No (sequential passes)
- **Order Sensitivity**: Higher (depends on initial ranking)
- **Best for**: Large document sets when only top-k matters

## Supported LLM Providers

The ranker supports any OpenAI-compatible API:

| Provider | Configuration |
|----------|---------------|
| OpenAI | Default (no `base_url` needed) |
| Groq | `base_url="https://api.groq.com/openai/v1"` |
| Together AI | `base_url="https://api.together.xyz/v1"` |
| Local (Ollama) | `base_url="http://localhost:11434/v1"` |
| vLLM | `base_url="http://localhost:8000/v1"` |

## Prompt Template

The ranker uses the following prompt structure:

**System Prompt:**
```
You are a ranking assistant that compares two passages in response to a query.
Respond ONLY with 'A' if Passage A is better, or 'B' if Passage B is better.
Please provide the response in a valid JSON format with the field name "selected_passage".
Example: {"selected_passage": "A"}
```

**User Prompt:**
```
Given a query {query}, which of the following two passages is more relevant to the query?
Passage A: {document1}
Passage B: {document2}
Output 'A' or 'B':
```

## Usage Examples

### Basic Usage

```python
from prp import PairwiseRankingPrompting

reranker = PairwiseRankingPrompting(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_key="your-api-key",
    base_url="https://api.groq.com/openai/v1"
)

query = "What are the benefits of regular exercise?"
documents = [
    "Exercise improves cardiovascular health.",
    "The Moon affects ocean tides.",
    "Regular workouts boost mental health.",
]

# Rerank with heapsort (recommended)
ranked = reranker.rerank(query, documents, method="heapsort")
print(ranked)
```

### With Token Tracking

```python
reranker = PairwiseRankingPrompting(model_name="gpt-4o-mini")

# Perform multiple reranking operations
results1 = reranker.rerank(query1, docs1)
results2 = reranker.rerank(query2, docs2)

# Check total tokens consumed
print(f"Total tokens used: {reranker.get_token_count()}")
```

### Custom Completion Parameters

```python
reranker = PairwiseRankingPrompting(
    model_name="gpt-4o-mini",
    completion_kwargs={
        "temperature": 0.0,
        "max_tokens": 50,
    }
)
```

### Top-K Results with Sliding Window

```python
# Get only top 5 documents with 5 passes
top_5 = reranker.rerank(
    query,
    documents,
    method="sliding_k",
    top_k=5,
    sliding_k_passes=5
)
```

## Best Practices

1. **Choose the right method**:
   - Small document sets (< 20): Use `allpairs` for best quality
   - Medium sets (20-100): Use `heapsort` for good balance
   - Large sets (> 100): Use `sliding_k` for efficiency

2. **Set temperature to 0** for deterministic results:
   ```python
   completion_kwargs={"temperature": 0.0}
   ```

3. **Monitor token usage** when working with large document sets:
   ```python
   print(f"Tokens: {reranker.get_token_count()}")
   ```

4. **Use appropriate top_k** to avoid unnecessary comparisons:
   ```python
   reranker.rerank(query, docs, method="sliding_k", top_k=10)
   ```
