import argparse
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from milvus_haystack import MilvusDocumentStore
from rankers.config import IndexingConfig, load_config
from rankers.dataloader import Dataloader


def main(config_path: str):
    """Index documents into Milvus vector database using Instructor embeddings."""
    config = load_config(path=Path(config_path), model=IndexingConfig)

    # Load dataset
    data_loader = Dataloader(dataset_name=config.dataset.name)
    dataset = data_loader.load()
    corpus = dataset.corpus
    documents = [
        Document(
            content=text_dict["text"],
            meta={"doc_id": str(document_id)},
        )
        for document_id, text_dict in corpus.items()
    ]

    # Initialize document store
    document_store = MilvusDocumentStore(
        connection_args={
            "uri": config.milvus.connection_uri,
            "token": config.milvus.connection_token,
        },
        **config.milvus.document_store_kwargs,
    )

    # Initialize embedder
    embedder = SentenceTransformersDocumentEmbedder(model=config.embedding.model, **config.embedding.model_kwargs)
    embedder.warm_up()

    # Create indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", DocumentWriter(document_store))
    indexing_pipeline.connect("embedder", "writer")

    # Index documents
    indexing_pipeline.run({"embedder": {"documents": documents}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents into Milvus vector database using Instructor embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        default="indexing_config.yaml",
        help="Path to the configuration file. Example: 'indexing_config.yaml'",
    )

    args = parser.parse_args()

    main(config_path=args.config_path)
