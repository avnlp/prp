# Pairwise Ranking Prompting (PRP)

This repository provides a implementation of **Pairwise Ranking Prompting (PRP)** technique from [Large Language Models are Effective Text Rankers with Pairwise Ranking
Prompting](https://arxiv.org/pdf/2306.17563).

The provide implementaion for the three PRP-based ranking strategies highlighted in the paper:

1. **`all_pair`**

- We enumerate all pairs and perform a global aggregation to generate a score for each document.
- Intuitively, if the LLM consistently prefers `Document-A` over `Document-B`, `Document-A` gets one point.
- When the LLM is not sure by producing conflicting or irrelevant results (for the generation API), each document gets half a point. There might be ties for the aggregated scores, in which case we fall back to initial ranking.
- `PRP-all_pair` is highly insensitive to input ordering. It essentially ranks documents with win ratio.
- The clear drawback is its costly $O(N^2)$ calls to LLM APIs, where N is the number of documents to be ranked for each query.

2. **`heapsort`**

- We use the pairwise preferences from the LLM as a comparator with HeapSort.
- `PRP-heapsort` favors lower computation complexity than `PRP-allpair` while also being large insensitive to input orders.
- This method makes $O(N \log N)$ calls to LLM APIs.

3. **`sliding_k`**

- We use a sliding window that starts at the bottom of the initial ranking, compares pairs of documents, and swap document pairs with a stride of 1. One sliding window is similar to one pass of BubbleSort.
- Since we want to optimize for the `top-k` results, we perform the sliding window operation `k` times.
- Complexity is linear in $O(N)$ times `k`.
- `PRP-sliding-k` has favorable time complexity but has high dependency on input order.

Key Features of the PairwiseRankingPrompting implementation:

- **Structured Generation with Pydantic Validation**: These ranker leverage structured generation and robust Pydantic validation to ensure accurate zero-shot ranking, even on smaller LLMs.
- **Efficient Sorting Algorithms**: The `PairwisePromptingRanker` utilizes efficient sorting methods (Heapsort and Bubblesort) to speed up inference.
- **Evaluation Toolkit**: We provide a custom Evaluator and Dataloader for evaluating rankers on standard metrics (NDCG, MAP, Recall, Precision) at various cutoffs. The Dataloader efficiently loads and processes datasets using the `ir_datasets` library.

## Usage Example

```python
from prp import PairwiseRankingPrompting

# Initialize PRP-based reranker
reranker = PairwiseRankingPrompting(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_key="API_KEY",
    base_url="BASE_URL"
)

query = "What are the benefits of regular exercise?"
documents = [
    "Engaging in regular exercise enhances cardiovascular fitness and helps regulate blood pressure.",
    "The Moon is Earth's only natural satellite and affects tides on the planet.",
    "Going to the gym regularly can help build muscle strength and improve mental health.",
]

# Use sliding_k to pick top 2 items
results_sliding_k = reranker.rerank(
    query, documents, method="sliding_k", top_k=2
)

# Rank all documents with heapsort
results_heapsort = reranker.rerank(
    query, documents, method="heapsort"
)

# Rank all documents with all_pair
results_allpairs = reranker.rerank(
    query, documents, method="all_pair"
)
```

## Evaluation

We evaluated the `PairwiseRankingPrompting` using pipelines built with the Haystack framework.

The evaluation was conducted on the following datasets: FIQA, SciFact, NFCorpus, TREC-19, and TREC-20.

The Mistral, Phi-3, and Llama-3 models were used with the `PairwiseRankingPrompting` ranker.

The evaluation pipelines can be found in the [pipelines](src/rankers/pipelines) directory.

**Evaluation Results**:  

We report the `NDCG@10` scores for each dataset and method in the table below:

| **Model**     | **Ranker**    | **FiQA**   | **SciFACT** | **NFCorpus** | **TREC-19** | **TREC-19** |
| ------------- | -------------- | ---------- | ----------- | ------------ | ----------- | ----------- |
| Mistral       | PRP-sliding_k | 0.4664     | 0.6847    | 0.4261   | 0.7062  | 0.6860  |
| Mistral       | PRP-heapsort  | 0.4672     | 0.6860    | 0.4311   | 0.7134  | 0.6875  |
| Mistral       | PRP-allpair   | 0.4676     | 0.6860    | 0.4312   | 0.7186  | 0.6987  |
| Phi-3         | PRP-sliding_k | 0.4704     | 0.6980    | 0.4365   | 0.7202  | 0.7140  |
| Phi-3         | PRP-heapsort  | 0.4712     | 0.6990    | 0.4385   | 0.7226  | 0.7154  |
| Phi-3         | PRP-allpair   | 0.4714     | 0.7028    | 0.4386   | 0.7228  | 0.7167  |
| Llama-3       | PRP-heapsort  | 0.4764     | 0.7765    | 0.4423   | 0.7508  | 0.7637  |
| Llama-3       | PRP-sliding_k | 0.4793     | 0.7852    | 0.4503   | 0.7511  | 0.7642  |
| Llama-3       | PRP-allpair   | **0.4992** | **0.7912** | **0.4658** | **0.7623** | **0.7671** |

- We find that `PRP-allpair` performed the best across all datasets.
- `PRP-sliding_k` and `PRP-heapsort` perform similarly across all datasets.
- The `PRP-allpair` with the Llama-3 model performed the best across all datasets.
