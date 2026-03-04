# Configuration

Documentation for configuration schemas and YAML-based configuration loading.

## Configuration Classes

### PairwiseRankerConfig

Configuration for the PairwiseRankingPrompting ranker.

```python
class PairwiseRankerConfig(BaseModel):
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    client_kwargs: dict[str, Any] = {}
    completion_kwargs: dict[str, Any] = {}
    method: PairwiseMethod = PairwiseMethod.HEAPSORT
    top_k: int = 10
    sliding_k_passes: int = 10
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | Required | LLM model identifier |
| `api_key` | `str \| None` | `None` | API key (defaults to env variable) |
| `base_url` | `str \| None` | `None` | Custom API endpoint URL |
| `client_kwargs` | `dict` | `{}` | OpenAI client settings |
| `completion_kwargs` | `dict` | `{}` | Completion parameters (temperature, max_tokens) |
| `method` | `PairwiseMethod` | `HEAPSORT` | Ranking algorithm |
| `top_k` | `int` | `10` | Max documents to return |
| `sliding_k_passes` | `int` | `10` | Passes for sliding_k method |

### PairwiseMethod

Enum for available ranking methods.

```python
class PairwiseMethod(str, Enum):
    HEAPSORT = "heapsort"
    ALLPAIRS = "allpairs"
    SLIDING_K = "sliding_k"
```

### DatasetConfig

Configuration for dataset loading.

```python
class DatasetConfig(BaseModel):
    name: str  # ir_datasets format, e.g., "beir/fiqa/train"
```

### EmbeddingConfig

Configuration for embedding models.

```python
class EmbeddingConfig(BaseModel):
    model: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: dict[str, Any] = {}
```

### MilvusConfig

Configuration for Milvus vector database.

```python
class MilvusConfig(BaseModel):
    connection_uri: str  # e.g., "http://localhost:19530"
    connection_token: str
    document_store_kwargs: dict[str, Any] = {}
```

### EvaluationConfig

Configuration for evaluation metrics.

```python
class EvaluationConfig(BaseModel):
    cutoff_values: list[int] = [1, 3, 5, 10]
    ignore_identical_ids: bool = False
    decimal_precision: int = 4
    metrics_to_compute: list[str] = ["ndcg", "map", "precision", "recall"]
```

### RetrievalConfig

Configuration for document retrieval.

```python
class RetrievalConfig(BaseModel):
    filters: dict[str, Any] = {}
    documents_to_retrieve: int = 25
```

### PairwiseRankingPromptingConfig

Full pipeline configuration combining all components.

```python
class PairwiseRankingPromptingConfig(BaseModel):
    dataset: DatasetConfig
    prp: PairwiseRankerConfig
    embedding: EmbeddingConfig
    milvus: MilvusConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig = EvaluationConfig()
```

## YAML Configuration

### Loading Configuration

```python
from prp.config import load_config, PairwiseRankingPromptingConfig

config = load_config("config.yaml", PairwiseRankingPromptingConfig)
```

### Example YAML Files

#### Basic Ranker Configuration

```yaml
# ranker_config.yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct"
api_key: null  # Uses OPENAI_API_KEY env variable
base_url: "https://api.groq.com/openai/v1"
method: heapsort
top_k: 10
completion_kwargs:
  temperature: 0.0
  max_tokens: 50
```

#### Full Pipeline Configuration

```yaml
# pipeline_config.yaml
dataset:
  name: "beir/scifact/test"

prp:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  base_url: "https://api.groq.com/openai/v1"
  method: heapsort
  top_k: 10
  completion_kwargs:
    temperature: 0.0

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  model_kwargs: {}

milvus:
  connection_uri: "http://localhost:19530"
  connection_token: "your-token"
  document_store_kwargs:
    collection_name: "scifact_collection"

retrieval:
  documents_to_retrieve: 100
  filters: {}

evaluation:
  cutoff_values: [1, 3, 5, 10]
  ignore_identical_ids: true
  decimal_precision: 4
  metrics_to_compute:
    - ndcg
    - map
    - recall
    - precision
```

#### Evaluation-Only Configuration

```yaml
# eval_config.yaml
cutoff_values: [5, 10, 20]
ignore_identical_ids: true
decimal_precision: 3
metrics_to_compute:
  - ndcg
  - map
```

## Usage Examples

### Using YAML Configuration

```python
from prp.config import load_config, PairwiseRankerConfig
from prp import PairwiseRankingPrompting

# Load configuration
config = load_config("ranker_config.yaml", PairwiseRankerConfig)

# Create ranker from config
ranker = PairwiseRankingPrompting(
    model_name=config.model_name,
    api_key=config.api_key,
    base_url=config.base_url,
    completion_kwargs=config.completion_kwargs,
)

# Use ranker
results = ranker.rerank(
    query="example query",
    documents=["doc1", "doc2", "doc3"],
    method=config.method.value,
    top_k=config.top_k,
)
```

### Programmatic Configuration

```python
from prp.config import PairwiseRankerConfig, PairwiseMethod

config = PairwiseRankerConfig(
    model_name="gpt-4o-mini",
    method=PairwiseMethod.ALLPAIRS,
    top_k=5,
    completion_kwargs={"temperature": 0.0},
)

# Validate and use
print(f"Using method: {config.method.value}")
print(f"Top-k: {config.top_k}")
```

### Environment Variables

For sensitive data like API keys, use environment variables:

```yaml
# config.yaml
model_name: "gpt-4o-mini"
api_key: null  # Will use OPENAI_API_KEY environment variable
```

```bash
export OPENAI_API_KEY="your-api-key"
python your_script.py
```

## Configuration Patterns

### Multiple Environments

```yaml
# config_dev.yaml
prp:
  model_name: "gpt-3.5-turbo"  # Cheaper for development
  method: sliding_k
  top_k: 5

# config_prod.yaml
prp:
  model_name: "gpt-4o"  # Better quality for production
  method: allpairs
  top_k: 10
```

### Method-Specific Configurations

```yaml
# For quick evaluation (sliding_k)
prp:
  method: sliding_k
  top_k: 10
  sliding_k_passes: 10

# For best quality (allpairs)
prp:
  method: allpairs
  top_k: 20

# For balanced approach (heapsort)
prp:
  method: heapsort
  top_k: 15
```

## Validation

All configuration classes use Pydantic for validation:

```python
from pydantic import ValidationError
from prp.config import PairwiseRankerConfig

try:
    config = PairwiseRankerConfig(
        model_name="gpt-4o-mini",
        top_k=-1,  # Invalid: must be > 0
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

Common validation rules:
- `top_k` must be greater than 0
- `sliding_k_passes` must be greater than 0
- `method` must be a valid `PairwiseMethod` value
- `model_name` is required
