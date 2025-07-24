from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from prp.dataloader.dataloader import Dataloader, Dataset


@pytest.fixture
def mock_ir_dataset():
    """Fixture to mock the `ir_datasets.load` function.

    This fixture patches `ir_datasets.load` to return a mock dataset that can be used for testing.

    Yields:
        MagicMock: A mock dataset with iterable methods for documents, queries, and relevance judgments.
    """
    with patch("prp.dataloader.dataloader.load_dataset") as mock_load_function:
        mock_dataset_instance = MagicMock()
        mock_load_function.return_value = mock_dataset_instance
        yield mock_dataset_instance


class TestDataset:
    """Test suite for the Dataloader and Dataset classes."""

    def test_dataloader_load(self, mock_ir_dataset):
        """Tests that the Dataloader correctly loads a dataset and structures it properly.

        Args:
            mock_ir_dataset (MagicMock): The mocked dataset fixture.
        """
        # Setup mock data
        mock_ir_dataset.docs_iter.return_value = [
            MagicMock(doc_id="doc1", text="Text 1"),
            MagicMock(doc_id="doc2", text="Text 2"),
        ]
        mock_ir_dataset.queries_iter.return_value = [
            MagicMock(query_id="query1", text="Query 1"),
            MagicMock(query_id="query2", text="Query 2"),
        ]
        mock_ir_dataset.qrels_iter.return_value = [
            MagicMock(query_id="query1", doc_id="doc1", relevance=1),
            MagicMock(query_id="query1", doc_id="doc2", relevance=0),
            MagicMock(query_id="query2", doc_id="doc2", relevance=2),
        ]

        # Load dataset
        loaded_dataset = Dataloader("mock_dataset").load()

        # Validate structure
        assert isinstance(loaded_dataset, Dataset)
        assert len(loaded_dataset.corpus) == 2
        assert all("text" in document for document in loaded_dataset.corpus.values())

        # Validate relevance aggregation
        assert loaded_dataset.relevance_judgments["query1"]["doc1"] == 1

    @pytest.mark.parametrize("dataset_name", ["invalid/dataset", "nonexistent"])
    def test_dataloader_invalid_dataset(self, dataset_name):
        """Tests that the Dataloader raises an error when an invalid dataset is provided.

        Args:
            dataset_name (str): The name of the dataset that does not exist.
        """
        with patch("prp.dataloader.dataloader.load_dataset", side_effect=ValueError("Dataset not found")):
            with pytest.raises(ValueError, match="Dataset not found"):
                Dataloader(dataset_name).load()

    def test_dataloader_missing_text_field(self, mock_ir_dataset):
        """Tests that the Dataloader raises an error if a document lacks a 'text' field.

        Args:
            mock_ir_dataset (MagicMock): The mocked dataset fixture.
        """
        # Create a dummy document without a 'text' attribute using SimpleNamespace
        document_without_text = SimpleNamespace(doc_id="doc1")

        # Set the docs_iter to return our dummy document
        mock_ir_dataset.docs_iter.return_value = [document_without_text]

        # Expect that accessing document_without_text.text will raise an AttributeError
        with pytest.raises(AttributeError):
            Dataloader("missing_text_dataset").load()

    def test_dataset_type_contracts(self):
        """Tests that the Dataset class maintains correct data type contracts."""
        # Create a sample dataset
        sample_dataset = Dataset(
            corpus={"doc1": {"text": "..."}},
            queries={"query1": "..."},
            relevance_judgments={"query1": {"doc1": 1}},
        )

        # Validate types
        assert isinstance(sample_dataset.corpus, dict)
        assert isinstance(sample_dataset.queries, dict)
        assert isinstance(sample_dataset.relevance_judgments, dict)
        assert all(
            isinstance(relevance_value, int)
            for query_relevance in sample_dataset.relevance_judgments.values()
            for relevance_value in query_relevance.values()
        )
