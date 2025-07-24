from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    """Container for evaluation metrics with validation.

    Attributes:
        ndcg (Dict[str, float]): NDCG scores keyed by metric string (e.g., 'NDCG@1').
        map (Dict[str, float]): MAP scores keyed by metric string (e.g., 'MAP@1').
        recall (Dict[str, float]): Recall scores keyed by metric string (e.g., 'RECALL@1').
        precision (Dict[str, float]): Precision scores keyed by metric string (e.g., 'PRECISION@1').
    """

    ndcg: dict[str, float]
    map: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]

    def __post_init__(self) -> None:
        """Validate that all metric dictionaries use the same cutoff values.

        Raises:
            ValueError: If inconsistent cutoff values are detected among the metrics.
        """
        # For each metric, extract the cutoff values (ignoring the metric name)
        cutoff_sets = []
        for metric in (self.ndcg, self.map, self.recall, self.precision):
            cutoffs = {key.split("@")[1] for key in metric.keys()}
            cutoff_sets.append(cutoffs)
        if not all(x == cutoff_sets[0] for x in cutoff_sets):
            msg = "All metrics must use the same k-values."
            raise ValueError(msg)
