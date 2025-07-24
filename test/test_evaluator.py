import logging
from unittest.mock import MagicMock

import pytest

from prp.evaluation.evaluator import Evaluator
from prp.evaluation.evaluator_params import EvaluatorParams


@pytest.fixture
def sample_relevance_judgments():
    """Fixture providing sample relevance judgments for testing.

    Returns:
        dict: A dictionary of query-document relevance judgments.
    """
    return {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 1, "d4": 0}}


@pytest.fixture
def sample_run_results():
    """Fixture providing sample run results for testing.

    Returns:
        dict: A dictionary of query-document run results with scores.
    """
    return {"q1": {"d1": 1.0, "d2": 0.5}, "q2": {"d3": 0.9, "d4": 0.8}}


@pytest.fixture
def evaluator_config():
    """Fixture creating a default EvaluatorParams for testing.

    Returns:
        EvaluatorParams: A configuration object with predefined settings.
    """
    return EvaluatorParams(
        cutoff_values=(1, 3, 5, 10),  # Specify different cutoff points for metrics
        ignore_identical_ids=True,  # Remove query-document pairs with identical IDs
        decimal_precision=4,  # Set precision for metric calculations
        metrics_to_compute=("ndcg", "map", "recall", "precision"),  # Metrics to evaluate
    )


class TestEvaluator:
    """Test suite for the Evaluator class, covering initialization and metric computation."""

    def test_initialization_with_valid_data(self, sample_relevance_judgments, sample_run_results):
        """Test that Evaluator can be initialized with valid relevance judgments and run results.

        Args:
            sample_relevance_judgments (dict): Predefined relevance judgments.
            sample_run_results (dict): Predefined run results.
        """
        evaluator = Evaluator(sample_relevance_judgments, sample_run_results)
        assert evaluator is not None
        assert evaluator.config is not None

    def test_initialization_with_empty_relevance_judgments_raises_error(self):
        """Test that initializing Evaluator with empty relevance judgments raises a ValueError."""
        with pytest.raises(ValueError):
            Evaluator({}, {"q1": {"d1": 1.0}})

    def test_initialization_with_empty_run_results_raises_error(self):
        """Test that initializing Evaluator with empty run results raises a ValueError."""
        with pytest.raises(ValueError):
            Evaluator({"q1": {"d1": 1}}, {})

    def test_invalid_cutoff_values_raises_error(self):
        """Test that initializing Evaluator with invalid cutoff values (zero) raises a ValueError."""
        with pytest.raises(ValueError):
            Evaluator({"q1": {"d1": 1}}, {"q1": {"d1": 1.0}}, EvaluatorParams(cutoff_values=(0, 5)))

    def test_warning_when_no_common_queries(self, caplog):
        """Test that a warning is logged when there are no common queries between relevance judgments and run results.

        Args:
            caplog: pytest fixture for capturing log messages.
        """
        with caplog.at_level(logging.WARNING):
            Evaluator({"q1": {"d1": 1}}, {"q2": {"d1": 1.0}})
        assert "No common queries" in caplog.text

    def test_filter_identical_ids_enabled(self):
        """Test that identical IDs are removed when ignore_identical_ids is True."""
        run_results = {"q1": {"q1": 0.9, "d2": 0.8}, "q2": {"d3": 0.7}}
        evaluator = Evaluator({"q1": {}, "q2": {}}, run_results, EvaluatorParams(ignore_identical_ids=True))
        evaluator._filter_identical_ids()
        assert evaluator.run_results == {"q1": {"d2": 0.8}, "q2": {"d3": 0.7}}

    def test_filter_identical_ids_disabled(self):
        """Test that identical IDs are not removed when ignore_identical_ids is False."""
        run_results = {"q1": {"q1": 0.9, "d2": 0.8}}
        evaluator = Evaluator({"q1": {}}, run_results, EvaluatorParams(ignore_identical_ids=False))
        evaluator._filter_identical_ids()
        assert evaluator.run_results == run_results

    def test_identical_ids_logging(self, caplog):
        """Test logging of removed identical IDs during evaluation.

        Args:
            caplog: pytest fixture for capturing log messages.
        """
        run_results = {"q1": {"q1": 0.9, "d2": 0.8}}
        with caplog.at_level(logging.INFO):
            evaluator = Evaluator({"q1": {}}, run_results, EvaluatorParams(ignore_identical_ids=True))
            evaluator.evaluate()
        assert "Removed 1 query-document pairs" in caplog.text

    def test_compute_base_metrics_with_valid_metrics(self, monkeypatch):
        """Test computation of base metrics with valid metric selections.

        Args:
            monkeypatch: pytest fixture for modifying behavior during testing.
        """
        # Create a mock evaluator to simulate metric computation
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"q1": {"ndcg_cut_5": 0.5}}
        monkeypatch.setattr("prp.evaluation.evaluator.RelevanceEvaluator", lambda *args, **kwargs: mock_evaluator)

        evaluator = Evaluator({"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("ndcg",), cutoff_values=(5,)))
        raw_scores = evaluator._compute_base_metrics()
        assert raw_scores == {"q1": {"ndcg_cut_5": 0.5}}

    def test_compute_base_metrics_with_invalid_metric_raises_error(self):
        """Test that specifying an invalid metric raises a ValueError."""
        with pytest.raises(ValueError):
            Evaluator({"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("invalid",)))

    def test_compute_average_metrics(self):
        """Test computation of average metrics across queries."""
        raw_scores = {"q1": {"ndcg_cut_5": 0.5, "map_cut_5": 0.6}, "q2": {"ndcg_cut_5": 0.7, "map_cut_5": 0.8}}
        evaluator = Evaluator(
            {"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("ndcg", "map"), cutoff_values=(5,))
        )

        averaged = evaluator._compute_average_metrics(raw_scores)
        assert averaged["ndcg"]["NDCG@5"] == 0.6
        assert averaged["map"]["MAP@5"] == 0.7

    def test_compute_average_metrics_with_missing_data(self):
        """Test computation of average metrics when some queries have missing data."""
        raw_scores = {"q1": {"ndcg_cut_5": 0.5}, "q2": {}}
        evaluator = Evaluator(
            {"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("ndcg", "map"), cutoff_values=(5,))
        )

        averaged = evaluator._compute_average_metrics(raw_scores)
        assert averaged["ndcg"]["NDCG@5"] == 0.25

    def test_compute_average_metrics_zero_queries(self, caplog):
        """Test computation of average metrics when no valid queries are present.

        Args:
            caplog: pytest fixture for capturing log messages.
        """
        evaluator = Evaluator({"q1": {}}, {"q1": {}}, EvaluatorParams(metrics_to_compute=("ndcg",), cutoff_values=(5,)))
        averaged = evaluator._compute_average_metrics({})
        assert "No valid queries" in caplog.text
        assert averaged["ndcg"]["NDCG@5"] == 0.0

    def test_full_evaluation_workflow(self, monkeypatch, sample_relevance_judgments, sample_run_results):
        """Test the complete evaluation workflow with all metrics.

        Args:
            monkeypatch: pytest fixture for modifying behavior during testing.
            sample_relevance_judgments (dict): Predefined relevance judgments.
            sample_run_results (dict): Predefined run results.
        """
        # Mock the RelevanceEvaluator to return predefined scores
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "q1": {"ndcg_cut_1": 0.8, "map_cut_1": 0.7, "recall_1": 0.6, "P_1": 0.5},
            "q2": {"ndcg_cut_1": 0.7, "map_cut_1": 0.6, "recall_1": 0.5, "P_1": 0.4},
        }
        monkeypatch.setattr("prp.evaluation.evaluator.RelevanceEvaluator", lambda *args, **kwargs: mock_evaluator)

        # Create evaluator with specific configuration
        evaluator = Evaluator(
            sample_relevance_judgments,
            sample_run_results,
            EvaluatorParams(cutoff_values=(1,), metrics_to_compute=("ndcg", "map", "recall", "precision")),
        )
        metrics = evaluator.evaluate()

        # Verify averaged metrics
        assert metrics.ndcg["NDCG@1"] == 0.75
        assert metrics.map["MAP@1"] == 0.65
        assert metrics.recall["RECALL@1"] == 0.55
        assert metrics.precision["PRECISION@1"] == 0.45

    def test_access_metrics_before_evaluation_raises_error(self):
        """Test that accessing metrics before evaluation raises a RuntimeError."""
        evaluator = Evaluator({"q1": {}}, {"q1": {}})
        with pytest.raises(RuntimeError):
            _ = evaluator.evaluation_metrics

    def test_evaluation_logging(self, caplog, sample_relevance_judgments, sample_run_results):
        """Test logging during the evaluation process.

        Args:
            caplog: pytest fixture for capturing log messages.
            sample_relevance_judgments (dict): Predefined relevance judgments.
            sample_run_results (dict): Predefined run results.
        """
        with caplog.at_level(logging.INFO):
            evaluator = Evaluator(sample_relevance_judgments, sample_run_results)
            evaluator.evaluate()
        assert "Evaluation completed successfully" in caplog.text
        assert "Evaluation Metrics:" in caplog.text
