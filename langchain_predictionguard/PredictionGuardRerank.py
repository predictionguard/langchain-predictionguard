from copy import deepcopy
import logging
from typing import Any, Optional, Union, Dict, Sequence, List
import yaml

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import ConfigDict, model_validator

logger = logging.getLogger(__name__)

class PredictionGuardRerank(BaseDocumentCompressor):
    """Document compressor that uses the PredictionGuard API."""

    client: Any = None  #: :meta private:
    """Prediction Guard Client"""

    model: Optional[str] = "bge-reranker-v2-m3"
    """Model name to use."""

    predictionguard_api_key: Optional[str] = None
    """Prediction Guard API key."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the api_key and python package exists in environment."""
        pg_api_key = get_from_dict_or_env(
            values, "predictionguard_api_key", "PREDICTIONGUARD_API_KEY"
        )

        try:
            from predictionguard import PredictionGuard

            values["client"] = PredictionGuard(
                api_key=pg_api_key,
            )

        except ImportError:
            raise ImportError(
                "Could not import predictionguard python package. "
                "Please install it with `pip install predictionguard`."
            )

        return values



    def _document_to_str(
        self,
        document: Union[str, Document, dict],
    ) -> str:
        if isinstance(document, Document):
            return document.page_content
        elif isinstance(document, dict):
            filtered_dict = document

            return yaml.dump(filtered_dict, sort_keys=False)
        else:
            return document

    def rerank(
        self,
        query: str,
        documents: Sequence[Union[str, Document, dict]],
        *,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            model: The model used for reranking.
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
        """  # noqa: E501
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [self._document_to_str(doc) for doc in documents]
        print(docs)
        model = model or self.model
        results = self.client.rerank.create(
            query=query,
            documents=docs,
            model=model,
        )
        result_dicts = []
        for res in results["results"]:
            result_dicts.append(
                {"index": res["index"], "relevance_score": res["relevance_score"]}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Prediction Guard's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed