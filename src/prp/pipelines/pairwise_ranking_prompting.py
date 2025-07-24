import argparse
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever
from tqdm import tqdm

from prp import Dataloader, Evaluator, EvaluatorParams, PairwiseRankingPrompting
from prp.config import PairwiseRankingPromptingConfig, load_config


def main(config_path: str):
    """Run a pipeline evaluating the quality of the retrieved documents after using the Pairwise LLM Ranker.

    The pipeline consists of:
    1. Loading dataset with ir_datasets format
    2. Initializing Milvus document store and embedding retriever
    3. Creating text embedding pipeline with Instructor model
    4. Reranking documents using PairwiseRankingPrompting
    5. Evaluating results with standard IR metrics
    """
    config = load_config(Path(config_path), PairwiseRankingPromptingConfig)

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
        top_k=config.retrieval.documents_to_retrieve,
        filters=config.retrieval.filters,
    )

    # Initialize text embedder
    text_embedder = SentenceTransformersTextEmbedder(model=config.embedding.model, **config.embedding.model_kwargs)

    # Initialize ranker
    llm_ranker = PairwiseRankingPrompting(
        model_name=config.prp.model_name,
        api_key=config.prp.api_key,
        base_url=config.prp.base_url,
        client_kwargs=config.prp.client_kwargs,
        completion_kwargs=config.prp.completion_kwargs,
    )

    # Create and connect pipeline
    embedding_pipeline = Pipeline()
    embedding_pipeline.add_component(instance=text_embedder, name="text_embedder")
    embedding_pipeline.add_component(instance=milvus_retriever, name="embedding_retriever")
    embedding_pipeline.add_component(instance=llm_ranker, name="ranker")
    embedding_pipeline.connect("text_embedder", "embedding_retriever")

    # Process each query
    all_query_results: dict[str, dict[str, float]] = {}
    for query_id, query in tqdm(queries.items()):
        # Retrieve initial candidate documents
        pipeline_output = embedding_pipeline.run({"text_embedder": {"text": query}})
        retrieved_docs = pipeline_output["embedding_retriever"]["documents"]

        # Extract contents and original IDs
        contents = [doc.content for doc in retrieved_docs]
        doc_ids = [doc.id for doc in retrieved_docs]

        # Rerank by content strings
        sorted_contents = llm_ranker.rerank(
            query,
            contents,
            method=config.prp.method,
            top_k=config.prp.top_k,
            sliding_k_passes=config.prp.sliding_k_passes,
        )

        # Map sorted contents back to original document indices (preserving duplicates)
        index_map: dict[str, list[int]] = {}
        for idx, content in enumerate(contents):
            index_map.setdefault(content, []).append(idx)

        sorted_indices: list[int] = []
        for content in sorted_contents:
            idx = index_map[content].pop(0)
            sorted_indices.append(idx)

        # Assign descending scores (higher rank -> higher score)
        num = len(sorted_indices)
        document_scores: dict[str, float] = {}
        for rank, idx in enumerate(sorted_indices):
            orig_id = doc_ids[idx]
            score = float(num - rank)
            document_scores[orig_id] = score

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
        description="Evaluation pipeline for Pairwise LLM Ranker in information retrieval tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    main(args.config_path)
