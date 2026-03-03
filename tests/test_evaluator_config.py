"""Tests for EvaluatorConfig."""

import pytest

from prp.evaluation.evaluator_config import EvaluatorConfig


class TestEvaluatorConfig:
    """Test suite for the EvaluatorConfig dataclass."""

    def test_valid_config_initialization(self):
        """Test that EvaluatorConfig initializes with valid values."""
        config = EvaluatorConfig(
            cutoff_values=(1, 3, 5, 10),
            ignore_identical_ids=True,
            decimal_precision=4,
            metrics_to_compute=("ndcg", "map", "recall", "precision"),
        )
        assert config.cutoff_values == (1, 3, 5, 10)
        assert config.ignore_identical_ids is True
        assert config.decimal_precision == 4

    def test_default_values(self):
        """Test that EvaluatorConfig has sensible defaults."""
        config = EvaluatorConfig()
        assert config.cutoff_values == (1, 3, 5, 10)
        assert config.ignore_identical_ids is True
        assert config.decimal_precision == 4
        assert config.metrics_to_compute == ("ndcg", "map", "recall", "precision")

    def test_empty_cutoff_values_raises_error(self):
        """Test that empty cutoff_values raises ValueError."""
        with pytest.raises(ValueError, match="non-empty tuple"):
            EvaluatorConfig(cutoff_values=())

    def test_non_tuple_cutoff_values_raises_error(self):
        """Test that non-tuple cutoff_values raises ValueError."""
        with pytest.raises(ValueError, match="non-empty tuple"):
            EvaluatorConfig(cutoff_values=[1, 3, 5])  # type: ignore

    def test_negative_cutoff_value_raises_error(self):
        """Test that negative cutoff values raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            EvaluatorConfig(cutoff_values=(-1, 3))

    def test_zero_cutoff_value_raises_error(self):
        """Test that zero cutoff value raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            EvaluatorConfig(cutoff_values=(0, 5))

    def test_negative_decimal_precision_raises_error(self):
        """Test that negative decimal_precision raises ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            EvaluatorConfig(decimal_precision=-1)

    def test_non_integer_decimal_precision_raises_error(self):
        """Test that non-integer decimal_precision raises ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            EvaluatorConfig(decimal_precision=0.5)  # type: ignore

    def test_empty_metrics_to_compute_raises_error(self):
        """Test that empty metrics_to_compute raises ValueError."""
        with pytest.raises(ValueError, match="non-empty tuple"):
            EvaluatorConfig(metrics_to_compute=())

    def test_non_tuple_metrics_to_compute_raises_error(self):
        """Test that non-tuple metrics_to_compute raises ValueError."""
        with pytest.raises(ValueError, match="non-empty tuple"):
            EvaluatorConfig(metrics_to_compute=["ndcg"])  # type: ignore

    def test_invalid_metric_name_raises_error(self):
        """Test that invalid metric names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid metric"):
            EvaluatorConfig(metrics_to_compute=("ndcg", "invalid_metric"))
