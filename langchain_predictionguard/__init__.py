"""Create controlled and compliant AI systems with PredictionGuard and LangChain"""
import warnings

from .chat_prediction_guard import ChatPredictionGuard as ChatPredictionGuard
from .prediction_guard import PredictionGuard as PredictionGuard
from .prediction_guard_embeddings import (
    PredictionGuardEmbeddings as PredictionGuardEmbeddings,
)
from .prediction_guard_rerank import PredictionGuardRerank as PredictionGuardRerank

__version__ = "0.4.0"

warnings.warn(
    "langchain-predictionguard is deprecated and no longer maintained. "
    "Some features may be broken or missing. Use langchain-openai or "
    "langchain-anthropic pointed at the Prediction Guard API instead. See "
    "https://github.com/predictionguard/langchain-predictionguard#readme "
    "for migration details.",
    FutureWarning,
    stacklevel=2,
)