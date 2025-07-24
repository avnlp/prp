import pytest

from prp.evaluation.evaluation_metrics import EvaluationMetrics
from prp.evaluation.evaluator import Evaluator
from prp.evaluation.evaluator_params import EvaluatorParams


class TestEvaluationMetrics:
    """Test suite for the EvaluationMetrics class, ensuring correct initialization and data handling."""

    def test_evaluation_metrics_validation(self):
        """Test that EvaluationMetrics raises a ValueError for metrics with different lengths.

        Ensures that metrics passed to EvaluationMetrics must have consistent keys across
        different metric types to maintain data integrity.
        """
        with pytest.raises(ValueError):
            EvaluationMetrics(
                ndcg={"NDCG@5": 0.5}, map={"MAP@10": 0.6}, recall={"RECALL@5": 0.7}, precision={"PRECISION@5": 0.8}
            )

    def test_metrics_extraction(self):
        """Test that EvaluationMetrics correctly stores and allows extraction of individual metric values.

        Verifies that metrics can be accessed by their specific keys after initialization,
        and that the values are preserved exactly as input.
        """
        metrics = EvaluationMetrics(
            ndcg={"NDCG@5": 0.1235},
            map={"MAP@5": 0.4354},
            recall={"RECALL@5": 0.6542},
            precision={"PRECISION@5": 0.2345},
        )

        # Verify each metric type can be correctly retrieved
        assert metrics.ndcg["NDCG@5"] == 0.1235
        assert metrics.map["MAP@5"] == 0.4354
        assert metrics.recall["RECALL@5"] == 0.6542
        assert metrics.precision["PRECISION@5"] == 0.2345

    def test_decimal_precision(self):
        """Test that metric computations respect decimal precision configuration.

        Ensures that average metrics are correctly rounded to the specified number of
        decimal places during computation.
        """
        # Simulate raw scores from query evaluations
        raw_scores = {"q1": {"ndcg_cut_5": 0.12345}}

        # Create evaluator with default configuration
        evaluator = Evaluator(
            {"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("ndcg", "map"), cutoff_values=(5,))
        )

        # Compute average metrics
        averaged = evaluator._compute_average_metrics(raw_scores)

        # Verify rounding to 4 decimal places
        assert averaged["ndcg"]["NDCG@5"] == 0.1235
