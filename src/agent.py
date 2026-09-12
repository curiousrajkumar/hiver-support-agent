"""Orchestration Agent for AppleSupport AI Support System.

Combines Intent Classification, BM25 Grounding Retrieval,
Grounded Reply Drafting, and Multi-factor Escalation Engine.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.escalation import EscalationDecision, EscalationEngine
from src.intent_classifier import IntentClassifier
from src.llm_client import LLMClient
from src.reply_generator import GroundedReplyGenerator
from src.retrieval import SupportRetrievalEngine

logger = logging.getLogger(__name__)


class AppleSupportAgent:
    """Full AI Customer Support Pipeline Agent."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retrieval_engine: Optional[SupportRetrievalEngine] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        reply_generator: Optional[GroundedReplyGenerator] = None,
        escalation_engine: Optional[EscalationEngine] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.retrieval = retrieval_engine or SupportRetrievalEngine()
        self.classifier = intent_classifier or IntentClassifier(llm_client=self.llm)
        self.generator = reply_generator or GroundedReplyGenerator(llm_client=self.llm)
        self.escalation = escalation_engine or EscalationEngine()

        # Ensure retrieval index is ready
        self.retrieval.initialize()

    def process(self, customer_tweet: str) -> EscalationDecision:
        """Process incoming customer tweet through full pipeline."""
        # 1. Intent Classification
        prediction = self.classifier.classify(customer_tweet)

        # 2. Historical Retrieval Grounding
        retrieved_pairs = self.retrieval.retrieve(customer_tweet, top_k=3)

        # 3. Grounded Reply Drafting
        drafted_reply = self.generator.draft_reply(
            customer_tweet=customer_tweet,
            intent=prediction.intent,
            retrieved_pairs=retrieved_pairs,
        )

        # 4. Multi-factor Auto vs Escalate Decision
        decision = self.escalation.evaluate(
            customer_tweet=customer_tweet,
            prediction=prediction,
            retrieved_pairs=retrieved_pairs,
            drafted_reply=drafted_reply,
        )

        return decision
