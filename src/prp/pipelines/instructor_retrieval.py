import argparse
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from rankers import Dataloader, Evaluator, EvaluatorParams
from rankers.config import RetrievalConfig, load_config
from tqdm import tqdm


def main(config_path: str):
    """Run a pipeline evaluating the quality of the retrieved documents.

    The pipeline consists of:
    1. Loading configuration from YAML
    2. Loading dataset with ir_datasets format
    3. Initializing Milvus document store and embedding retriever
    4. Creating text embedding pipeline with Instructor model
    5. Evaluating results with standard IR metrics

    Args:
        config_path: Path to the YAML configuration file.
    """
    # Load configuration
    config = load_config(Path(config_path), RetrievalConfig)

    # Load dataset
    dataloader = Dataloader(config.dataset.name)
    dataset = dataloader.load()
    queries = dataset.queries
    relevance_judgments = dataset.relevance_judgments

    # Initialize document store and retriever
    milvus_document_store = MilvusDocumentStore(
        connection_args={
            "uri": config.milvus.connection_uri,
            "token": config.milvus.connection_token,
        },
        **config.milvus.document_store_kwargs,
    )
    milvus_retriever = MilvusEmbeddingRetriever(
        document_store=milvus_document_store,
        top_k=config.documents_to_retrieve,
        filters=config.filters,
    )

    # Initialize text embedder
    text_embedder = SentenceTransformersTextEmbedder(
        model=config.embedding.model,
        **config.embedding.model_kwargs,
    )

    # Create and connect pipeline
    embedding_pipeline = Pipeline()
    embedding_pipeline.add_component(instance=text_embedder, name="text_embedder")
    embedding_pipeline.add_component(instance=milvus_retriever, name="embedding_retriever")
    embedding_pipeline.connect("text_embedder", "embedding_retriever")

    # Process each query
    all_query_results = {}
    for query_id, query in tqdm(queries.items()):
        pipeline_output = embedding_pipeline.run({"text_embedder": {"text": query}})
        ranked_documents = pipeline_output["embedding_retriever"]["documents"]
        document_scores = {}
        for document in ranked_documents:
            document_scores[document.meta["doc_id"]] = document.score
        all_query_results[query_id] = document_scores

    # Evaluate results
    evaluation_config = EvaluatorParams(
        cutoff_values=tuple(config.evaluation.cutoff_values),
        ignore_identical_ids=config.evaluation.ignore_identical_ids,
        decimal_precision=config.evaluation.decimal_precision,
        metrics_to_compute=tuple(config.evaluation.metrics_to_compute),
    )
    evaluator = Evaluator(
        relevance_judgments=relevance_judgments,
        run_results=all_query_results,
        config=evaluation_config,
    )
    evaluation_metrics = evaluator.evaluate()
    print(evaluation_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Run the Instructor-based retrieval pipeline with YAML configuration")
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the YAML configuration file. If not provided, looks for "
            "'instructor_retrieval_config.yaml' in the current directory."
        ),
    )
    args = parser.parse_args()

    main(config_path=args.config)
