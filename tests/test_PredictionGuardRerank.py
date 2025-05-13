"""Test Prediction Guard API wrapper"""

from langchain_predictionguard import PredictionGuardRerank


def test_langchain_cohere_rerank_documents() -> None:
    rerank = PredictionGuardRerank(model="bge-reranker-v2-m3")
    test_query = "Test query"
    test_documents = [
        "This is a test document.",
        "Another test document."
    ]
    results = rerank.rerank(query=test_query, documents=test_documents)
    assert len(results) == 2