import json
from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
from openai import OpenAI

from prp.prompts import SYSTEM_PROMPT, USER_PROMPT, PairwiseRankingResponse


class PairwiseRankingPrompting:
    """Implements pairwise document reranking with position bias mitigation.

    Attributes:
        model_name (str): Name of the LLM to use for comparisons.
        client (OpenAI): Initialized OpenAI API client.
        total_tokens (int): Cumulative tokens consumed across API calls.
        default_completion_params (dict): Default parameters for chat completion calls.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the LLM and ranking parameters.

        Args:
            model_name: Name of the LLM to use for comparisons.
            api_key: API key for authentication (defaults to environment variable).
            base_url: Custom API endpoint URL, if any.
            client_kwargs: Additional keyword arguments passed to the OpenAI client.
            completion_kwargs: Default parameters for completion calls (e.g., temperature).
        """
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
        if client_kwargs:
            client_args.update(client_kwargs)

        self.client = OpenAI(**client_args)
        self.model_name = model_name
        self.total_tokens = 0
        self.default_completion_params = completion_kwargs or {}

    def get_token_count(self) -> int:
        """Retrieve the total number of tokens used.

        Returns:
            int: Cumulative token count.
        """
        return self.total_tokens

    def rerank(
        self,
        query: str,
        documents: list[str],
        method: str = "allpairs",
        top_k: int | None = None,
        sliding_k_passes: int | None = None,
    ) -> list[str]:
        """Rerank a list of documents by relevance to the query.

        Args:
            query: User search query string.
            documents: List of document texts to be reranked.
            method: Ranking algorithm to use: "allpairs" (O(n^2)),
                    "heapsort" (O(n log n)), or "sliding_k" (O(k * n)).
            top_k: If specified, limits results to the top k documents.
            sliding_k_passes: For method "sliding_k", the number of passes to do
                            (defaults to top_k if provided, else 10).

        Returns:
            List[str]: Documents sorted by descending relevance.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        if not documents:
            return []

        if method == "heapsort":
            sorted_docs = self._heapsort_rerank(query, documents)
        elif method == "allpairs":
            sorted_docs = self._allpairs_rerank(query, documents)
        elif method == "sliding_k":
            if sliding_k_passes is not None:
                num_passes = sliding_k_passes
            elif top_k is not None:
                num_passes = top_k
            else:
                num_passes = 10
            sorted_docs = self._sliding_k_rerank(query, documents, num_passes)
        else:
            msg = f"Unsupported method '{method}'. Choose 'allpairs', 'heapsort', or 'sliding_k'."
            raise ValueError(msg)

        return sorted_docs[:top_k] if top_k is not None else sorted_docs

    def _compare_pair(self, query: str, doc_a: str, doc_b: str) -> str:
        """Send a pairwise comparison request to the LLM and returns the preferred label.

        Args:
            query: Search query string.
            doc_a: Text of the first document.
            doc_b: Text of the second document.

        Returns:
            str: "A" if doc_a is preferred, "B" if doc_b is preferred.
        """
        try:
            user_content = USER_PROMPT.format(query=query, document1=doc_a, document2=doc_b)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                **self.default_completion_params,
            )
            # Aggregate token usage
            usage = getattr(response, "usage", None)
            if usage and getattr(usage, "total_tokens", None):
                self.total_tokens += usage.total_tokens

            response_content = response.choices[0].message.content.strip()
            try:
                result = json.loads(response_content)
                return PairwiseRankingResponse(**result).selected_passage
            except (ValueError, TypeError, KeyError):
                # Fallback: first character A or B
                return response_content[:1].upper() if response_content[:1] in ("A", "B") else "A"
        except Exception:
            # On API failure, default to first document
            return "A"

    def _compare_with_bias_mitigation(
        self, query: str, doc_index_a: int, doc_index_b: int, documents: list[str]
    ) -> tuple[float, float]:
        """Mitigates LLM position bias by comparing documents in both orders.

        Args:
            query: Search query string.
            doc_index_a: Index of the first document in the list.
            doc_index_b: Index of the second document in the list.
            documents: Full list of document texts.

        Returns:
            Tuple[float, float]: Scores for doc_a and doc_b (0.0, 0.5, or 1.0).
        """
        # Compare A vs B and B vs A
        pref_ab = self._compare_pair(query, documents[doc_index_a], documents[doc_index_b])
        pref_ba = self._compare_pair(query, documents[doc_index_b], documents[doc_index_a])

        if pref_ab == "A" and pref_ba == "B":
            return 1.0, 0.0
        if pref_ab == "B" and pref_ba == "A":
            return 0.0, 1.0
        # Tie or inconsistent
        return 0.5, 0.5

    def _build_max_heap(self, indices: list[int], heap_size: int, compare_fn: Callable[[int, int], bool]) -> None:
        """Convert a list of indices into a max-heap in place.

        Args:
            indices: List of document indices to heapify.
            heap_size: Number of elements in the heap.
            compare_fn: Function to compare two indices (True if first should come first).
        """
        for root in range(heap_size // 2 - 1, -1, -1):
            self._sift_down(indices, root, heap_size, compare_fn)

    def _sift_down(self, indices: list[int], root: int, heap_size: int, compare_fn: Callable[[int, int], bool]) -> None:
        """Maintains the max-heap property by sifting a root element down.

        Args:
            indices: Heap list of document indices.
            root: Current root position to sift down.
            heap_size: Active size of the heap.
            compare_fn: Function to compare two indices.
        """
        while True:
            left = 2 * root + 1
            right = 2 * root + 2
            largest = root

            if left < heap_size and compare_fn(indices[left], indices[largest]):
                largest = left
            if right < heap_size and compare_fn(indices[right], indices[largest]):
                largest = right

            if largest == root:
                break

            indices[root], indices[largest] = indices[largest], indices[root]
            root = largest

    def _heapsort_rerank(self, query: str, documents: list[str]) -> list[str]:
        """Rerank documents using an efficient O(n log n) heapsort-based approach.

        Args:
            query: Search query string.
            documents: List of document texts.

        Returns:
            List[str]: Documents ordered by descending relevance.
        """
        n = len(documents)
        if n < 2:
            return documents.copy()

        indices = list(range(n))

        # Define comparison for heap: True if i precedes j
        def compare_indices(i: int, j: int) -> bool:
            score_i, score_j = self._compare_with_bias_mitigation(query, i, j, documents)
            if score_i != score_j:
                return score_i > score_j
            return i < j  # stable tie-break

        # Build heap and perform sort
        self._build_max_heap(indices, n, compare_indices)
        for end in range(n - 1, 0, -1):
            indices[0], indices[end] = indices[end], indices[0]
            self._sift_down(indices, 0, end, compare_indices)

        # Return documents in descending relevance
        return [documents[i] for i in reversed(indices)]

    def _sliding_k_rerank(self, query: str, documents: list[str], num_passes: int) -> list[str]:
        """Rerank documents using the PRP-Sliding-K approach with K passes.

        Args:
            query: Search query string.
            documents: List of document texts.
            num_passes: Number of sliding window passes to perform.

        Returns:
            List[str]: Documents ordered by descending relevance.
        """
        n = len(documents)
        if n <= 1:
            return documents.copy()

        # Copy document list for in-place modification
        docs = documents.copy()

        # Perform K passes of sliding window comparisons
        for _ in range(num_passes):
            # Traverse from bottom to top (last element to first)
            for i in range(n - 2, -1, -1):
                # Compare adjacent documents
                preference = self._compare_pair(query, docs[i], docs[i + 1])
                # Swap if second document is preferred
                if preference == "B":
                    docs[i], docs[i + 1] = docs[i + 1], docs[i]

        return docs

    def _allpairs_rerank(self, query: str, documents: list[str]) -> list[str]:
        """Rerank documents by performing all-pairs comparisons (O(n^2)).

        Args:
            query: Search query string.
            documents: List of document texts.

        Returns:
            List[str]: Documents ordered by descending aggregate preference score.
        """
        n = len(documents)
        if n < 2:
            return documents.copy()

        scores = np.zeros(n)
        for i, j in combinations(range(n), 2):
            score_i, score_j = self._compare_with_bias_mitigation(query, i, j, documents)
            scores[i] += score_i
            scores[j] += score_j

        ranked_indices = list(np.argsort(-scores))
        return [documents[i] for i in ranked_indices]
