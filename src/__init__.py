"""AppleSupport AI Agent Package."""

from src.agent import AppleSupportAgent
from src.baselines import SimpleBaselineAgent, TrivialBaselineAgent
from src.escalation import EscalationDecision, EscalationEngine
from src.intent_classifier import IntentClassifier, IntentPrediction
from src.llm_client import LLMClient
from src.reply_generator import GroundedReplyGenerator
from src.retrieval import SupportRetrievalEngine

__all__ = [
    "AppleSupportAgent",
    "TrivialBaselineAgent",
    "SimpleBaselineAgent",
    "EscalationEngine",
    "EscalationDecision",
    "IntentClassifier",
    "IntentPrediction",
    "LLMClient",
    "GroundedReplyGenerator",
    "SupportRetrievalEngine",
]
