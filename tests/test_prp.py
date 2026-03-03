"""Tests for PairwiseRankingPrompting implementation."""

import json
from types import SimpleNamespace

import pytest

from prp.prp import PairwiseRankingPrompting


class MockResponse:
    """Mock response class to simulate chat completions."""

    def __init__(self, content, total_tokens=None):
        """Initialize the mock response with content and optional token count.

        Args:
            content: The content of the response.
            total_tokens: Optional token count for the response.
        """
        # Simulate response format
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        self.choices = [choice]

        # Optional token usage info
        if total_tokens is not None:
            self.usage = SimpleNamespace(total_tokens=total_tokens)


class MockChat:
    """Mock chat client that returns a predefined response."""

    def __init__(self, response):
        """Initialize the mock chat client with a predefined response.

        Args:
            response: Predefined response object to return from create method.
        """
        self._response = response

    @property
    def completions(self):
        """Return the mock response object."""
        return self

    def create(self, **kwargs):
        """Return the mock response object."""
        return self._response


@pytest.fixture
def prp_instance():
    """Fixture to create a fresh PairwiseRankingPrompting instance.

    Returns:
        PairwiseRankingPrompting: An instance with dummy model and API info.
    """
    return PairwiseRankingPrompting(
        model_name="test-model", api_key="llm_api_key", base_url="llm_base_url"
    )


class TestPairwiseRankingPrompting:
    """Unit tests for the PairwiseRankingPrompting class."""

    def test_get_token_count_initial(self, prp_instance):
        """Test that initial token count is zero."""
        assert prp_instance.get_token_count() == 0

    def test_rerank_empty(self):
        """Test reranking with an empty document list returns empty list."""
        pr = PairwiseRankingPrompting(
            model_name="m", api_key="llm_api_key", base_url="llm_base_url"
        )
        assert pr.rerank("q", [], method="allpairs") == []

    def test_rerank_invalid_method(self, prp_instance):
        """Test that reranking with an invalid method raises ValueError."""
        with pytest.raises(ValueError) as exc:
            prp_instance.rerank("q", ["d1"], method="invalid")
        assert "Unsupported method" in str(exc.value)

    def test_rerank_calls_allpairs_and_slices(self, monkeypatch, prp_instance):
        """Test rerank with top_k slicing from the allpairs method."""
        docs = ["a", "b", "c"]
        monkeypatch.setattr(
            prp_instance, "_allpairs_rerank", lambda query, documents: ["X", "Y", "Z"]
        )
        res = prp_instance.rerank("q", docs, method="allpairs", top_k=2)
        assert res == ["X", "Y"]

    @pytest.mark.parametrize(
        "resp_content,expected,tokens",
        [
            (json.dumps({"selected_passage": "B"}), "B", 5),
            ("A extra text", "A", None),
            ("unknown", "A", None),
        ],
    )
    def test_compare_pair_various(self, prp_instance, resp_content, expected, tokens):
        """Test _compare_pair with various mock responses.

        Args:
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
            resp_content (str): Simulated model output.
            expected (str): Expected returned passage ('A' or 'B').
            tokens (int or None): Simulated token usage.
        """
        resp = MockResponse(resp_content, total_tokens=tokens)
        prp_instance.client = SimpleNamespace(
            chat=SimpleNamespace(completions=MockChat(resp))
        )
        result = prp_instance._compare_pair("q", "d1", "d2")
        assert result == expected
        if tokens:
            assert prp_instance.get_token_count() == tokens

    def test_compare_pair_exception(self, prp_instance):
        """Test _compare_pair handles API exceptions gracefully.

        Returns 'A' by default when exceptions occur.

        Args:
            prp_instance: The PRP instance to test.
        """

        class BadChat:
            @property
            def completions(self):
                msg = "API down"
                raise RuntimeError(msg)

        prp_instance.client = SimpleNamespace(chat=BadChat())
        assert prp_instance._compare_pair("q", "d1", "d2") == "A"

    @pytest.mark.parametrize(
        "pref_ab,pref_ba,exp",
        [
            ("A", "B", (1.0, 0.0)),
            ("B", "A", (0.0, 1.0)),
            ("A", "A", (0.5, 0.5)),
        ],
    )
    def test_compare_with_bias(self, monkeypatch, prp_instance, pref_ab, pref_ba, exp):
        """Test bias mitigation logic by comparing both (a,b) and (b,a) orderings.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for patching.
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
            pref_ab (str): Result when comparing A over B.
            pref_ba (str): Result when comparing B over A.
            exp (tuple): Expected final (score_a, score_b) pair.
        """
        calls = []

        def fake_compare(query, a, b):
            calls.append(1)
            return pref_ab if len(calls) == 1 else pref_ba

        monkeypatch.setattr(prp_instance, "_compare_pair", fake_compare)
        docs = ["a", "b"]
        scores = prp_instance._compare_with_bias_mitigation("q", 0, 1, docs)
        assert scores == exp

    def test_sift_down_simple(self):
        """Test that _sift_down correctly rearranges a heap with a custom comparator.

        Args:
            prp (PairwiseRankingPrompting): The PRP instance to test.
        """
        indices = [2, 3, 1]

        def compare(i, j):
            return i > j

        prp = PairwiseRankingPrompting(
            model_name="m", api_key="llm_api_key", base_url="llm_base_url"
        )
        prp._sift_down(indices, 0, len(indices), compare)
        index_value = 3
        assert indices[0] == index_value

    def test_heapsort_rerank_single(self):
        """Test heapsort rerank handles single document case correctly.

        Args:
            prp (PairwiseRankingPrompting): The PRP instance to test.
        """
        prp = PairwiseRankingPrompting(
            model_name="m", api_key="llm_api_key", base_url="llm_base_url"
        )
        docs = ["only"]
        assert prp._heapsort_rerank("q", docs) == docs

    def test_heapsort_rerank_two(self, monkeypatch, prp_instance):
        """Test heapsort rerank with two documents and mocked comparison.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for patching.
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
        """
        docs = ["a", "b"]

        def mock_compare(query, i, j, docs_list):
            return (1.0, 0.0) if i == 1 and j == 0 else (0.0, 1.0)

        monkeypatch.setattr(prp_instance, "_compare_with_bias_mitigation", mock_compare)
        sorted_docs = prp_instance._heapsort_rerank("q", docs)
        assert sorted_docs == ["b", "a"]

    def test_allpairs_rerank(self, monkeypatch, prp_instance):
        """Test all-pairs ranking preserves order based on consistent preferences.

        Args:
            monkeypatch: Pytest monkeypatch fixture for patching.
            prp_instance: The PRP instance to test.
        """
        docs = ["a", "b", "c"]

        def mock_compare(query, i, j, docs_list):
            return (1.0, 0.0) if i < j else (0.0, 1.0)

        monkeypatch.setattr(prp_instance, "_compare_with_bias_mitigation", mock_compare)
        ranked = prp_instance._allpairs_rerank("q", docs)
        assert ranked == ["a", "b", "c"]

    def test_sliding_k_rerank_single(self):
        """Test sliding-k rerank handles single document case correctly.

        Args:
            prp (PairwiseRankingPrompting): The PRP instance to test.
        """
        prp = PairwiseRankingPrompting(
            model_name="m", api_key="llm_api_key", base_url="llm_base_url"
        )
        assert prp._sliding_k_rerank("q", ["only"], num_passes=3) == ["only"]

    def test_sliding_k_rerank_two(self, monkeypatch, prp_instance):
        """Test sliding-k rerank with two documents and mocked compare_pair.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for patching.
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
        """
        docs = ["a", "b"]
        monkeypatch.setattr(prp_instance, "_compare_pair", lambda q, a, b: "B")
        result = prp_instance._sliding_k_rerank("q", docs, num_passes=1)
        assert result == ["b", "a"]

    def test_sliding_k_rerank_multi_pass(self, monkeypatch, prp_instance):
        """Test sliding-k rerank over multiple passes to fully sort documents.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for patching.
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
        """
        docs = ["a", "b", "c"]

        def fake_compare(q, a, b):
            return "B" if a < b else "A"

        monkeypatch.setattr(prp_instance, "_compare_pair", fake_compare)
        # One pass should move 'c' to the front
        result1 = prp_instance._sliding_k_rerank("q", docs, num_passes=1)
        assert result1 == ["c", "a", "b"]
        # Two passes should produce fully sorted descending order
        result2 = prp_instance._sliding_k_rerank("q", docs, num_passes=2)
        assert result2 == ["c", "b", "a"]

    def test_rerank_sliding_k_invocation_and_slicing(self, monkeypatch, prp_instance):
        """Test rerank method uses sliding_k backend and applies top_k slicing.

        Args:
            monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for patching.
            prp_instance (PairwiseRankingPrompting): The PRP instance to test.
        """
        docs = ["a", "b", "c"]
        monkeypatch.setattr(
            prp_instance, "_sliding_k_rerank", lambda q, d, num: ["X", "Y", "Z"]
        )
        res = prp_instance.rerank(
            "q", docs, method="sliding_k", top_k=2, sliding_k_passes=5
        )
        assert res == ["X", "Y"]

    def test_init_applies_client_kwargs(self, monkeypatch):
        """Test that client_kwargs are passed to the OpenAI client."""
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                chat=SimpleNamespace(completions=MockChat(MockResponse("{}")))
            )

        monkeypatch.setattr("prp.prp.OpenAI", fake_openai)
        PairwiseRankingPrompting(
            model_name="m",
            api_key="k",
            client_kwargs={"timeout": 30, "max_retries": 3},
        )
        assert captured["api_key"] == "k"
        assert captured["timeout"] == 30
        assert captured["max_retries"] == 3

    def test_rerank_heapsort_dispatch(self, monkeypatch, prp_instance):
        """Test that rerank dispatches to _heapsort_rerank when method='heapsort'."""
        monkeypatch.setattr(
            prp_instance, "_heapsort_rerank", lambda q, docs: ["H1", "H2"]
        )
        result = prp_instance.rerank("q", ["a", "b"], method="heapsort")
        assert result == ["H1", "H2"]

    def test_rerank_sliding_k_uses_top_k_when_no_sliding_k_passes(
        self, monkeypatch, prp_instance
    ):
        """Test sliding_k uses top_k as num_passes when sliding_k_passes is None."""
        seen = {}

        def fake_sliding(q, docs, num_passes):
            seen["num_passes"] = num_passes
            return docs

        monkeypatch.setattr(prp_instance, "_sliding_k_rerank", fake_sliding)

        prp_instance.rerank(
            "q", ["a", "b", "c"], method="sliding_k", top_k=3, sliding_k_passes=None
        )
        assert seen["num_passes"] == 3

    def test_rerank_sliding_k_defaults_to_10_passes(self, monkeypatch, prp_instance):
        """Test sliding_k defaults to 10 passes when top_k is not provided."""
        seen = {}

        def fake_sliding(q, docs, num_passes):
            seen["num_passes"] = num_passes
            return docs

        monkeypatch.setattr(prp_instance, "_sliding_k_rerank", fake_sliding)

        prp_instance.rerank(
            "q", ["a", "b"], method="sliding_k", top_k=None, sliding_k_passes=None
        )
        assert seen["num_passes"] == 10

    def test_sift_down_takes_right_child_branch(self):
        """Test that _sift_down correctly handles when right child is largest."""
        prp = PairwiseRankingPrompting(model_name="m", api_key="k", base_url="url")
        indices = [1, 2, 3]  # right child (3) should become largest

        def compare(i, j):
            return i > j

        prp._sift_down(indices, root=0, heap_size=len(indices), compare_fn=compare)
        assert indices[0] == 3

    def test_heapsort_tie_break_is_stable(self, monkeypatch, prp_instance):
        """Test that heapsort maintains stable order when scores tie."""
        monkeypatch.setattr(
            prp_instance,
            "_compare_with_bias_mitigation",
            lambda q, i, j, docs: (0.5, 0.5),
        )
        docs = ["d0", "d1", "d2"]
        result = prp_instance._heapsort_rerank("q", docs)
        assert result == docs

    def test_allpairs_rerank_single_document_returns_copy(self, prp_instance):
        """Test that _allpairs_rerank returns a copy for single document."""
        docs = ["only"]
        out = prp_instance._allpairs_rerank("q", docs)
        assert out == ["only"]
        assert out is not docs  # ensure it's a copy
