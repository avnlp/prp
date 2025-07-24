# This implementation is based on the Retrieval Evaluation from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py
# The code has been restructured with extensive Pydantic validation and error handling for metric edge cases.

import copy
import logging
from dataclasses import asdict

from pytrec_eval import RelevanceEvaluator

from prp.evaluation.evaluation_metrics import EvaluationMetrics
from prp.evaluation.evaluator_params import EvaluatorParams

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator:
    """Information Retrieval evaluation engine using pytrec_eval.

    This class evaluates IR system results against relevance judgments
    using the pytrec_eval library.

    Attributes:
        relevance_judgments (Dict[str, Dict[str, int]]): The ground truth relevance judgments.
        run_results (Dict[str, Dict[str, float]]): The results produced by an IR system.
        config (EvaluatorParams): Configuration for evaluation.
        _evaluation_metrics (EvaluationMetrics): Cached evaluation metrics after computation.
    """

    def __init__(
        self,
        relevance_judgments: dict[str, dict[str, int]],
        run_results: dict[str, dict[str, float]],
        config: EvaluatorParams | None = None,
    ) -> None:
        """Initialize the evaluator with relevance judgments, system results, and configuration.

        Args:
            relevance_judgments (Dict[str, Dict[str, int]]): Ground truth relevance judgments.
            run_results (Dict[str, Dict[str, float]]): System run results with scores.
            config (Optional[EvaluatorParams]): Configuration for evaluation. Defaults to EvaluatorParams() if None.

        Raises:
            ValueError: If input data is invalid.
        """
        self.config = config if config is not None else EvaluatorParams()
        self._validate_input_data(relevance_judgments, run_results, self.config.cutoff_values)

        self.relevance_judgments: dict[str, dict[str, int]] = copy.deepcopy(relevance_judgments)
        self.run_results: dict[str, dict[str, float]] = copy.deepcopy(run_results)
        self._evaluation_metrics: EvaluationMetrics | None = None

        logger.debug(
            "Initialized Evaluator with %d queries and %d relevance judgments",
            len(run_results),
            len(relevance_judgments),
        )

    def evaluate(self) -> EvaluationMetrics:
        """Evaluate the system results and compute evaluation metrics.

        Returns:
            EvaluationMetrics: An immutable container with averaged evaluation metrics.

        Raises:
            RuntimeError: If evaluation metrics have not been computed.
        """
        if self._evaluation_metrics is not None:
            return self._evaluation_metrics

        self._filter_identical_ids()
        raw_scores: dict[str, dict[str, float]] = self._compute_base_metrics()
        averaged_metrics: dict[str, dict[str, float]] = self._compute_average_metrics(raw_scores)
        self._evaluation_metrics = EvaluationMetrics(**averaged_metrics)

        logger.info("Evaluation completed successfully")
        self._log_evaluation_metrics()
        return self._evaluation_metrics

    def _validate_input_data(
        self,
        relevance_judgments: dict[str, dict[str, int]],
        run_results: dict[str, dict[str, float]],
        cutoff_values: tuple[int, ...],
    ) -> None:
        """Validate the input relevance judgments, system results, and cutoff values.

        Args:
            relevance_judgments (Dict[str, Dict[str, int]]): Ground truth relevance judgments.
            run_results (Dict[str, Dict[str, float]]): System run results with scores.
            cutoff_values (Tuple[int, ...]): Cutoff values for evaluation.

        Raises:
            ValueError: If relevance judgments or run results are empty, or if cutoff values are invalid.
        """
        if not relevance_judgments or not run_results:
            msg = "Relevance judgments and run results must be non-empty."
            raise ValueError(msg)

        if any(cutoff <= 0 for cutoff in cutoff_values):
            msg = "All cutoff values must be positive integers."
            raise ValueError(msg)

        common_queries = set(relevance_judgments) & set(run_results)
        if not common_queries:
            logger.warning("No common queries between relevance judgments and run results.")

    def _filter_identical_ids(self) -> None:
        """Filter out query-document pairs where the query ID is identical to the document ID."""
        if not self.config.ignore_identical_ids:
            return

        filtered_results: dict[str, dict[str, float]] = {}
        removed_pairs_count = 0

        for query_id, doc_scores in self.run_results.items():
            filtered_doc_scores = {doc_id: score for doc_id, score in doc_scores.items() if query_id != doc_id}
            removed_pairs_count += len(doc_scores) - len(filtered_doc_scores)
            filtered_results[query_id] = filtered_doc_scores

        self.run_results = filtered_results

        if removed_pairs_count > 0:
            logger.info(f"Removed {removed_pairs_count} query-document pairs with identical IDs.")

    def _compute_base_metrics(self) -> dict[str, dict[str, float]]:
        """Compute base evaluation metrics using pytrec_eval.

        Returns:
            Dict[str, Dict[str, float]]: Raw evaluation scores for each query.

        Raises:
            ValueError: If an undefined metric is requested in the configuration.
        """
        metric_commands: dict[str, str] = {
            "ndcg": f"ndcg_cut.{','.join(map(str, self.config.cutoff_values))}",
            "map": f"map_cut.{','.join(map(str, self.config.cutoff_values))}",
            "recall": f"recall.{','.join(map(str, self.config.cutoff_values))}",
            "precision": f"P.{','.join(map(str, self.config.cutoff_values))}",
        }

        selected_metric_commands = {
            metric: command for metric, command in metric_commands.items() if metric in self.config.metrics_to_compute
        }

        if len(selected_metric_commands) != len(self.config.metrics_to_compute):
            missing_metrics = set(self.config.metrics_to_compute) - set(selected_metric_commands.keys())
            msg = f"Undefined metrics requested: {missing_metrics}"
            raise ValueError(msg)

        evaluator = RelevanceEvaluator(self.relevance_judgments, set(selected_metric_commands.values()))
        return evaluator.evaluate(self.run_results)

    def _compute_average_metrics(self, raw_scores: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        # Initialize the accumulators for each metric and cutoff
        averaged_metrics: dict[str, dict[str, float]] = {metric: {} for metric in self.config.metrics_to_compute}
        for metric in self.config.metrics_to_compute:
            for cutoff in self.config.cutoff_values:
                metric_key = f"{metric.upper()}@{cutoff}"
                averaged_metrics[metric][metric_key] = 0.0

        num_queries = len(raw_scores)
        if num_queries == 0:
            logger.error("No valid queries found for averaging.")
            return averaged_metrics

        # Sum scores across queries using the correct key names.
        for query_scores in raw_scores.values():
            for metric in self.config.metrics_to_compute:
                for cutoff in self.config.cutoff_values:
                    if metric in ("ndcg", "map"):
                        score_key = f"{metric}_cut_{cutoff}"
                    elif metric == "precision":
                        score_key = f"P_{cutoff}"
                    else:
                        score_key = f"{metric}_{cutoff}"
                    averaged_metrics[metric][f"{metric.upper()}@{cutoff}"] += query_scores.get(score_key, 0.0)

        # Calculate averages and round the results.
        for metric in self.config.metrics_to_compute:
            for cutoff in self.config.cutoff_values:
                metric_key = f"{metric.upper()}@{cutoff}"
                averaged_metrics[metric][metric_key] = round(
                    averaged_metrics[metric][metric_key] / num_queries, self.config.decimal_precision
                )

        return averaged_metrics

    def _log_evaluation_metrics(self) -> None:
        """Log the computed evaluation metrics in a structured format."""
        logger.info("Evaluation Metrics:")
        for metric_name, metric_values in asdict(self._evaluation_metrics).items():
            logger.info("%s:", metric_name.upper())
            for key, value in metric_values.items():
                logger.info("  %s: %.4f", key, value)

    @property
    def evaluation_metrics(self) -> EvaluationMetrics:
        """Get the computed evaluation metrics.

        Returns:
            EvaluationMetrics: The evaluation metrics computed by the evaluator.

        Raises:
            RuntimeError: If evaluation has not been performed yet.
        """
        if self._evaluation_metrics is None:
            msg = "Evaluation metrics not computed. Please call evaluate() first."
            raise RuntimeError(msg)
        return self._evaluation_metrics
