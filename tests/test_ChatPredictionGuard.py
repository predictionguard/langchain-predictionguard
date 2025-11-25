"""Test Prediction Guard API wrapper."""

import os

import pytest

from langchain_predictionguard import PredictionGuard


def test_predictionguard_invoke() -> None:
    """Test valid call to prediction guard."""
    llm = PredictionGuard(model=os.environ["TEST_CHAT_MODEL"])  # type: ignore[call-arg]
    output = llm.invoke("Tell a joke.")
    assert isinstance(output, str)


def test_predictionguard_pii() -> None:
    llm = PredictionGuard(
        model=os.environ["TEST_CHAT_MODEL"],
        predictionguard_input={"pii": "block"},
        max_tokens=100,
        temperature=1.0,
    )

    messages = [
        "Hello, my name is John Doe and my SSN is 111-22-3333",
    ]

    with pytest.raises(ValueError, match=r"Could not make prediction. pii detected"):
        llm.invoke(messages)