from pydantic import BaseModel, Field, ValidationInfo, field_validator

SYSTEM_PROMPT = """You are a ranking assistant that compares two passages in response to a query.
Respond ONLY with 'A' if Passage A is better, or 'B' if Passage B is better.
Please provide the response in a valid JSON format with the field name "selected_passage".
Example: {"selected_passage": "A"}"""

USER_PROMPT = """Given a query {query}, which of the following two passages is more relevant to the query?
Passage A: {document1}
Passage B: {document2}
Output 'A' or 'B':"""


class PairwiseRankingResponse(BaseModel):
    """Represents validated output for pairwise document comparison selection.

    Validates that the selected passage identifier is either 'A' or 'B' in uppercase.
    Ensures the LLM's response matches the required format for reliable ranking.

    Attributes:
        selected_passage: The label of the more relevant passage. Must be uppercase
            'A' or 'B' without parentheses.

    Example:
        >>> PairwiseRankingResponse(selected_passage="A")
        PairwiseRankingResponse(selected_passage='A')

    Raises:
        ValueError: If selected_passage is not 'A' or 'B' in uppercase
    """

    selected_passage: str = Field(
        ...,
        pattern=r"^[AB]$",
        description=(
            "Identifier of the more relevant passage. Must be uppercase 'A' or 'B' "
            "without additional formatting. Example: 'A' for first passage, 'B' for second."
        ),
        examples=["A", "B"],
    )

    @field_validator("selected_passage")
    @classmethod
    def validate_identifier(cls, value: str, info: ValidationInfo) -> str:
        """Validate and normalize the passage selection identifier.

        Args:
            value: Raw value from LLM output
            info: Pydantic validation context (automatically populated)

        Returns:
            str: Normalized uppercase identifier if valid

        Raises:
            ValueError: For any input not matching 'A' or 'B' pattern

        Example:
            Valid inputs:
            - "A" → returns "A"
            - "B" → returns "B"

            Invalid inputs:
            - "a" → ValueError
            - "C" → ValueError
            - "(A)" → ValueError
        """
        normalized = value.strip().upper()
        if normalized not in {"A", "B"}:
            msg = f"Invalid selection: '{value}'. Must be uppercase 'A' or 'B' without parentheses."
            raise ValueError(msg)
        return normalized
