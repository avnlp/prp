"""Tests for prompt templates and response models."""

import pytest
from pydantic import ValidationError

from prp.prompts import PairwiseRankingResponse


class TestPairwiseRankingResponse:
    """Test suite for PairwiseRankingResponse validation."""

    @pytest.mark.parametrize("bad", ["(A)", "C", "A B", "", "AB", "1"])
    def test_invalid_selected_passage_raises_validation_error(self, bad):
        """Test that invalid selected_passage values raise ValidationError."""
        with pytest.raises(ValidationError):
            PairwiseRankingResponse(selected_passage=bad)

    @pytest.mark.parametrize("valid", ["A", "B"])
    def test_valid_selected_passage_accepted(self, valid):
        """Test that valid selected_passage values are accepted."""
        response = PairwiseRankingResponse(selected_passage=valid)
        assert response.selected_passage == valid
