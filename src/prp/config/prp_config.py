from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from prp.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)


class PairwiseMethod(str, Enum):
    """Available pairwise ranking methods (must match PairwiseRankingPrompting.rerank args)."""

    HEAPSORT = "heapsort"
    ALLPAIRS = "allpairs"
    SLIDING_K = "sliding_k"


class PairwiseRankerConfig(BaseModel):
    """Configuration for the PairwiseRankingPrompting ranker."""

    model_name: str = Field(
        ...,
        description=("LLM model name/path. E.g., 'meta-llama/Llama-3.1-8B-Instruct' or 'gpt-4o-mini'."),
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication (defaults to environment variable).",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API endpoint URL, if any.",
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM client settings (e.g., timeouts, proxies).",
    )
    completion_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Completion parameters (e.g., max_tokens, temperature).",
    )
    method: PairwiseMethod = Field(
        default=PairwiseMethod.HEAPSORT,
        description=("Ranking algorithm: heapsort (O(n log n)), allpairs (O(n²)), sliding_k (O(k·n))."),
    )
    top_k: int = Field(
        default=10,
        gt=0,
        description="Max # of docs to rerank/return (higher → slower).",
    )
    sliding_k_passes: int = Field(
        default=10,
        gt=0,
        description="Number of passes for sliding_k method (defaults to top_k if provided, else 10).",
    )


class PairwiseRankingPromptingConfig(BaseModel):
    """Configuration for the pairwise ranking pipeline."""

    dataset: DatasetConfig = Field(..., description="IR dataset identifier and split.")
    prp: PairwiseRankerConfig = Field(..., description="PairwiseRankingPrompting ranker settings.")
    embedding: EmbeddingConfig = Field(..., description="Embedding model settings.")
    milvus: MilvusConfig = Field(..., description="Milvus vector DB settings.")
    retrieval: RetrievalConfig = Field(..., description="Retrieval top-k & filters.")
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
        description="Evaluation metrics and cutoff settings.",
    )
