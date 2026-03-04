# Data Loading

Documentation for the Dataloader class and dataset structure.

## Dataloader

A unified interface to load benchmark datasets from the [ir_datasets](https://ir-datasets.com/) library.

### Constructor

```python
Dataloader(dataset_name: str)
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_name` | `str` | Yes | Dataset identifier in ir_datasets format (e.g., `"beir/fiqa/test"`) |

### Methods

#### `load()`

Load and process the dataset components.

```python
load() -> Dataset
```

##### Returns

`Dataset`: Container with corpus, queries, and relevance judgments.

## Dataset

Immutable container for information retrieval dataset components.

```python
@dataclass(frozen=True)
class Dataset:
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    relevance_judgments: dict[str, dict[str, int]]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `corpus` | `dict[str, dict[str, str]]` | Maps doc_id to document content (`{"text": "..."}`) |
| `queries` | `dict[str, str]` | Maps query_id to query text |
| `relevance_judgments` | `dict[str, dict[str, int]]` | Maps query_id to {doc_id: relevance_score} |

## Supported Datasets

### BEIR Benchmark

| Dataset | ID | Documents | Domain |
|---------|-----|-----------|--------|
| FIQA | `beir/fiqa/test` | ~57K | Financial Q&A |
| SciFact | `beir/scifact/test` | ~5K | Scientific claims |
| NFCorpus | `beir/nfcorpus/test` | ~3.6K | Nutrition/Medical |
| TREC-COVID | `beir/trec-covid` | ~171K | COVID-19 research |
| MS MARCO | `beir/msmarco/dev` | ~8.8M | Web passages |

### TREC Deep Learning

| Dataset | ID | Documents | Domain |
|---------|-----|-----------|--------|
| TREC-DL 2019 | `msmarco-passage/trec-dl-2019` | ~8.8M | Web passages |
| TREC-DL 2020 | `msmarco-passage/trec-dl-2020` | ~8.8M | Web passages |

### Other Collections

Any dataset available in [ir_datasets](https://ir-datasets.com/all.html) can be used. The format is typically:

```
collection/name/split
```

Examples:
- `antique/test`
- `natural-questions/dev`
- `clueweb12/b13/clef-ehealth`

## Usage Examples

### Basic Loading

```python
from prp.dataloader import Dataloader

# Load BEIR SciFact dataset
loader = Dataloader("beir/scifact/test")
dataset = loader.load()

# Access components
print(f"Corpus size: {len(dataset.corpus)}")
print(f"Number of queries: {len(dataset.queries)}")
print(f"Queries with relevance judgments: {len(dataset.relevance_judgments)}")
```

### Accessing Documents

```python
loader = Dataloader("beir/fiqa/test")
dataset = loader.load()

# Get a specific document
doc_id = list(dataset.corpus.keys())[0]
doc = dataset.corpus[doc_id]
print(f"Document text: {doc['text'][:200]}...")
```

### Accessing Queries and Relevance Judgments

```python
dataset = loader.load()

# Iterate over queries
for query_id, query_text in dataset.queries.items():
    print(f"Query {query_id}: {query_text}")

    # Get relevant documents for this query
    if query_id in dataset.relevance_judgments:
        qrels = dataset.relevance_judgments[query_id]
        for doc_id, relevance in qrels.items():
            print(f"  - Doc {doc_id}: relevance={relevance}")
```

### Integration with Ranker

```python
from prp import PairwiseRankingPrompting
from prp.dataloader import Dataloader

# Load dataset
loader = Dataloader("beir/scifact/test")
dataset = loader.load()

# Initialize ranker
ranker = PairwiseRankingPrompting(
    model_name="gpt-4o-mini",
    completion_kwargs={"temperature": 0.0}
)

# Process a query
query_id = list(dataset.queries.keys())[0]
query_text = dataset.queries[query_id]

# Get candidate documents (simplified - in practice use a retriever)
doc_ids = list(dataset.corpus.keys())[:20]
documents = [dataset.corpus[doc_id]["text"] for doc_id in doc_ids]

# Rerank
ranked_docs = ranker.rerank(query_text, documents, method="heapsort")
```

### Converting to Haystack Documents

```python
from haystack import Document
from prp.dataloader import Dataloader

loader = Dataloader("beir/nfcorpus/test")
dataset = loader.load()

# Convert corpus to Haystack Document objects
haystack_docs = [
    Document(
        id=doc_id,
        content=doc_data["text"],
    )
    for doc_id, doc_data in dataset.corpus.items()
]

print(f"Created {len(haystack_docs)} Haystack documents")
```

## Data Structures

### Corpus Format

```python
{
    "doc_id_1": {"text": "Document content here..."},
    "doc_id_2": {"text": "Another document..."},
    ...
}
```

### Queries Format

```python
{
    "query_id_1": "What is the capital of France?",
    "query_id_2": "How does photosynthesis work?",
    ...
}
```

### Relevance Judgments Format

```python
{
    "query_id_1": {
        "doc_id_1": 2,  # Highly relevant
        "doc_id_3": 1,  # Somewhat relevant
        "doc_id_5": 0,  # Not relevant
    },
    "query_id_2": {
        "doc_id_2": 1,
        "doc_id_4": 2,
    },
    ...
}
```

Relevance scores typically follow:
- `0`: Not relevant
- `1`: Marginally relevant
- `2`: Highly relevant
- `3`: Perfectly relevant (some datasets)
