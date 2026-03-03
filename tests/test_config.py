"""Tests for configuration classes."""

import pytest
from pydantic import ValidationError

from prp.config.base_config import (
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    MilvusConfig,
    RetrievalConfig,
)
from prp.config.config_loader import load_config
from prp.config.prp_config import (
    PairwiseMethod,
    PairwiseRankerConfig,
    PairwiseRankingPromptingConfig,
)


class TestBaseConfigs:
    """Tests for base configuration models."""

    def test_dataset_config_valid(self):
        """Test DatasetConfig with valid name."""
        cfg = DatasetConfig(name="beir/fiqa/train")
        assert cfg.name == "beir/fiqa/train"

    def test_embedding_config_strips_device_key(self):
        """Test that EmbeddingConfig removes device key from model_kwargs."""
        cfg = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda", "foo": 1},
        )
        assert "device" not in cfg.model_kwargs
        assert cfg.model_kwargs["foo"] == 1

    def test_embedding_config_without_device(self):
        """Test EmbeddingConfig without device in model_kwargs."""
        cfg = EmbeddingConfig(model="m", model_kwargs={"batch_size": 32})
        assert cfg.model_kwargs == {"batch_size": 32}

    def test_milvus_config_valid(self):
        """Test MilvusConfig with required fields."""
        cfg = MilvusConfig(
            connection_uri="http://localhost:19530", connection_token="token"
        )  # nosec B106
        assert cfg.connection_uri == "http://localhost:19530"
        assert cfg.connection_token == "token"  # nosec B105

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        cfg = EvaluationConfig()
        assert cfg.cutoff_values == [1, 3, 5, 10]
        assert cfg.ignore_identical_ids is False
        assert cfg.decimal_precision == 4
        assert cfg.metrics_to_compute == ["ndcg", "map", "precision", "recall"]

    @pytest.mark.parametrize("bad_precision", [-1, 7])
    def test_evaluation_config_decimal_precision_bounds(self, bad_precision):
        """Test that decimal_precision must be between 0 and 6."""
        with pytest.raises(ValidationError):
            EvaluationConfig(decimal_precision=bad_precision)

    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig default values."""
        cfg = RetrievalConfig()
        assert cfg.documents_to_retrieve == 25
        assert cfg.filters == {}

    def test_retrieval_config_documents_to_retrieve_must_be_positive(self):
        """Test that documents_to_retrieve must be > 0."""
        with pytest.raises(ValidationError):
            RetrievalConfig(documents_to_retrieve=0)


class TestPRPConfig:
    """Tests for PRP-specific configuration models."""

    def test_pairwise_method_enum_values(self):
        """Test PairwiseMethod enum has expected values."""
        assert PairwiseMethod.HEAPSORT.value == "heapsort"
        assert PairwiseMethod.ALLPAIRS.value == "allpairs"
        assert PairwiseMethod.SLIDING_K.value == "sliding_k"

    def test_pairwise_ranker_config_defaults(self):
        """Test PairwiseRankerConfig default values."""
        cfg = PairwiseRankerConfig(model_name="gpt-4o-mini")
        assert cfg.model_name == "gpt-4o-mini"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.method == PairwiseMethod.HEAPSORT
        assert cfg.top_k == 10
        assert cfg.sliding_k_passes == 10

    def test_pairwise_ranker_config_method_enum_parsing(self):
        """Test that method accepts string values."""
        cfg = PairwiseRankerConfig(model_name="m", method="allpairs")
        assert cfg.method == PairwiseMethod.ALLPAIRS

    def test_pairwise_ranker_config_rejects_invalid_method(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError):
            PairwiseRankerConfig(model_name="m", method="invalid_method")

    def test_pairwise_ranking_prompting_config_full(self):
        """Test full PairwiseRankingPromptingConfig construction."""
        cfg = PairwiseRankingPromptingConfig(
            dataset=DatasetConfig(name="beir/fiqa/train"),
            prp=PairwiseRankerConfig(model_name="test-model"),
            embedding=EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2"),
            milvus=MilvusConfig(
                connection_uri="http://localhost:19530", connection_token="token"
            ),  # nosec B106
            retrieval=RetrievalConfig(),
        )
        assert cfg.dataset.name == "beir/fiqa/train"
        assert cfg.evaluation.cutoff_values == [1, 3, 5, 10]  # default


class TestConfigLoader:
    """Tests for YAML config loading."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading a valid YAML configuration file."""
        p = tmp_path / "cfg.yml"
        p.write_text(
            """
dataset:
  name: beir/fiqa/train
prp:
  model_name: test-model
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
milvus:
  connection_uri: http://localhost:19530
  connection_token: token
retrieval:
  documents_to_retrieve: 25
""".strip()
        )

        cfg = load_config(p, PairwiseRankingPromptingConfig)
        assert cfg.dataset.name == "beir/fiqa/train"
        assert cfg.prp.model_name == "test-model"
        assert cfg.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_load_config_with_evaluation_overrides(self, tmp_path):
        """Test loading config with custom evaluation settings."""
        p = tmp_path / "cfg.yml"
        p.write_text(
            """
dataset:
  name: beir/scifact/test
prp:
  model_name: gpt-4o
  method: allpairs
  top_k: 5
embedding:
  model: m
milvus:
  connection_uri: uri
  connection_token: tok
retrieval: {}
evaluation:
  cutoff_values: [1, 5]
  decimal_precision: 2
""".strip()
        )

        cfg = load_config(p, PairwiseRankingPromptingConfig)
        assert cfg.prp.method == PairwiseMethod.ALLPAIRS
        assert cfg.evaluation.cutoff_values == [1, 5]
        assert cfg.evaluation.decimal_precision == 2
