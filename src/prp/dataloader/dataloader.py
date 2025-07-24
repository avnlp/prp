from ir_datasets import load as load_dataset

from prp.dataloader.dataset import Dataset


class Dataloader:
    """A data loader for information retrieval datasets using the ir_datasets library.

    This class handles loading of document corpora, queries, and query relevance judgments
    from datasets available in the ir_datasets package format.

    Attributes:
        dataset_name: String identifier of the dataset in ir_datasets format.
                        Example: 'beir/fiqa/train' for BEIR FiQA dataset train split.
    """

    def __init__(self, dataset_name: str) -> None:
        """Initialize the Dataloader with a specific dataset.

        Args:
            dataset_name: The ir_datasets identifier for the dataset and split to load.
                            Format is typically 'collection/name/split'.
                            Example: 'beir/fiqa/train'
        """
        self.dataset_name = dataset_name

    def load(self) -> Dataset:
        """Load and processes the dataset components.

        Returns:
            A Dataset object containing three elements:
            - document_corpus: Dictionary mapping document IDs to document content
                with 'text' and 'title' fields
            - query_texts: Dictionary mapping query IDs to query text
            - relevance_judgments: Dictionary mapping query IDs to dictionaries of
                relevant document IDs with their relevance scores

        Example:
            >>> loader = Dataloader('beir/fiqa/train')
            >>> dataset = loader.load()
            >>> corpus = dataset.corpus,
            >>> queries = dataset.queries
            >>> qrels = dataset.relevance_judgments
        """
        dataset = load_dataset(self.dataset_name)

        # Load corpus: {doc_id: {'text': text}}
        document_corpus: dict[str, dict[str, str]] = {}
        for document in dataset.docs_iter():
            document_corpus[document.doc_id] = {"text": document.text}

        # Load queries: {query_id: text}
        query_texts: dict[str, str] = {}
        for query in dataset.queries_iter():
            # Directly store text, no nested dict
            query_texts[query.query_id] = query.text

        # Load qrels: {query_id: {doc_id: relevance}}
        relevance_judgments: dict[str, dict[str, int]] = {}
        for judgment in dataset.qrels_iter():
            query_id = judgment.query_id
            doc_id = judgment.doc_id
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = {}
            relevance_judgments[query_id][doc_id] = judgment.relevance

        return Dataset(corpus=document_corpus, queries=query_texts, relevance_judgments=relevance_judgments)
