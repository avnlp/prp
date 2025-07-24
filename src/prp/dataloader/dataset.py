from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Container for information retrieval dataset components.

    Attributes:
        corpus: Dictionary mapping document IDs to document content.
            Each value is a dictionary with 'text' and 'title' keys.
        queries: Dictionary mapping query IDs to query text.
        relevance_judgments: Dictionary mapping query IDs to dictionaries of
            document IDs with their relevance scores.
    """

    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    relevance_judgments: dict[str, dict[str, int]]
