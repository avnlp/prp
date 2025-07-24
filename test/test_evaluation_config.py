import pytest

from prp.evaluation.evaluator_params import EvaluatorParams


class TestEvaluatorParams:
    """Test suite for the EvaluatorParams class, ensuring correct initialization and validation."""

    def test_valid_config_initialization(self):
        """Test that EvaluatorParams correctly initializes with custom configuration values.

        Verifies that all configuration parameters are correctly set during initialization
        and can be retrieved with the exact values provided.
        """
        config = EvaluatorParams(
            cutoff_values=(1, 3, 5, 10),
            ignore_identical_ids=True,
            decimal_precision=4,
            metrics_to_compute=("ndcg", "map", "recall", "precision"),
        )

        # Verify each configuration parameter is set correctly
        assert config.cutoff_values == (1, 3, 5, 10)
        assert config.ignore_identical_ids is True
        assert config.decimal_precision == 4
        assert config.metrics_to_compute == ("ndcg", "map", "recall", "precision")

    def test_invalid_cutoff_values_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for invalid cutoff values.

        Ensures that negative or zero cutoff values are not allowed,
        maintaining the integrity of metric computation.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(cutoff_values=(-1, 2))

    def test_invalid_decimal_precision_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for invalid decimal precision.

        Verifies that only integer values are accepted for decimal precision,
        preventing potential rounding issues.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(decimal_precision=0.5)

    def test_invalid_metrics_raises_error(self):
        """Test that EvaluatorParams raises a ValueError for unsupported metrics.

        Ensures that only predefined, valid metrics can be specified during configuration,
        preventing potential errors during evaluation.
        """
        with pytest.raises(ValueError):
            EvaluatorParams(metrics_to_compute=("invalid_metric",))

    def test_default_config_values(self):
        """Test that EvaluatorParams provides sensible default values when no parameters are specified.

        Verifies that:
        - Cutoff values are not empty
        - Ignore identical IDs is a boolean
        - Decimal precision is an integer
        - Metrics to compute are not empty
        """
        config = EvaluatorParams()

        # Verify default configuration parameters
        assert len(config.cutoff_values) > 0
        assert isinstance(config.ignore_identical_ids, bool)
        assert isinstance(config.decimal_precision, int)
        assert len(config.metrics_to_compute) > 0
