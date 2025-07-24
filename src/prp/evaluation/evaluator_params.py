from dataclasses import dataclass


@dataclass
class EvaluatorParams:
    """Configuration for evaluation parameters.

    Attributes:
        cutoff_values (Tuple[int, ...]): Cutoff values for evaluation metrics.
        ignore_identical_ids (bool): Whether to ignore results where query and document IDs are identical.
        decimal_precision (int): Number of decimal places for rounding metric values.
        metrics_to_compute (Tuple[str, ...]): Metrics to compute. Options: 'ndcg', 'map', 'recall', 'precision'.
    """

    cutoff_values: tuple[int, ...] = (1, 3, 5, 10)
    ignore_identical_ids: bool = True
    decimal_precision: int = 4
    metrics_to_compute: tuple[str, ...] = ("ndcg", "map", "recall", "precision")

    def __post_init__(self) -> None:
        """Validate evaluation parameters."""
        # Validate cutoff_values: non-empty and each value must be a positive integer.
        if not self.cutoff_values or not isinstance(self.cutoff_values, tuple):
            msg = "cutoff_values must be a non-empty tuple of positive integers."
            raise ValueError(msg)
        for cv in self.cutoff_values:
            if not isinstance(cv, int) or cv <= 0:
                msg = "Each cutoff value must be a positive integer."
                raise ValueError(msg)

        # Validate decimal_precision: must be a non-negative integer.
        if not isinstance(self.decimal_precision, int) or self.decimal_precision < 0:
            msg = "decimal_precision must be a non-negative integer."
            raise ValueError(msg)

        # Validate metrics_to_compute: non-empty tuple containing allowed metric names.
        allowed_metrics = {"ndcg", "map", "recall", "precision"}
        if not self.metrics_to_compute or not isinstance(self.metrics_to_compute, tuple):
            msg = "metrics_to_compute must be a non-empty tuple of allowed metric names."
            raise ValueError(msg)
        for metric in self.metrics_to_compute:
            if metric not in allowed_metrics:
                msg = f"Invalid metric: {metric}. Allowed metrics are {allowed_metrics}."
                raise ValueError(msg)
