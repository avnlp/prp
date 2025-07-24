from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(
        ...,
        description=("Dataset identifier in ir_datasets format. E.g., 'beir/fiqa/train'."),
    )


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model: str = Field(
        ...,
        description=("Embedding model name/path. E.g., 'sentence-transformers/all-MiniLM-L6-v2'."),
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for embedding model initialization.",
    )

    @model_validator(mode="after")
    def strip_device(self, values):
        """Remove stray device keys to avoid Haystack ComponentDevice errors."""
        values.model_kwargs.pop("device", None)
        return values


class MilvusConfig(BaseModel):
    """Configuration for the Milvus vector database."""

    connection_uri: str = Field(
        ...,
        description="Milvus server URI (e.g., 'http://localhost:19530')",
    )
    connection_token: str = Field(
        ...,
        description="Authentication token for Milvus",
    )
    document_store_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra parameters passed to MilvusDocumentStore",
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and settings."""

    cutoff_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10],
        description="Cutoff levels for metrics (e.g., NDCG@k).",
    )
    ignore_identical_ids: bool = Field(
        default=False,
        description=("Whether to exclude docs with the same ID as the query (prevents leakage)."),
    )
    decimal_precision: int = Field(
        default=4,
        ge=0,
        le=6,
        description="Decimal precision for reported metrics.",
    )
    metrics_to_compute: list[Literal["ndcg", "map", "precision", "recall"]] = Field(
        default_factory=lambda: ["ndcg", "map", "precision", "recall"],
        description="List of metrics to compute.",
    )


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline."""

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level filters for retrieval",
    )
    documents_to_retrieve: int = Field(
        default=25,
        gt=0,
        description="Number of documents to retrieve per query",
    )
