"""Multi-factor Escalation Engine for AppleSupport AI Agent.

Evaluates security keywords, hardware hazard terms, sentiment risk,
intent ambiguity, and retrieval similarity thresholds to decide: auto vs. escalate.
Outputs structured decision object.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import yaml

from src.intent_classifier import IntentPrediction
from src.retrieval import RetrievedPair

logger = logging.getLogger(__name__)


@dataclass
class EscalationDecision:
    intent: str
    action: str  # "auto" | "escalate"
    reason: str
    drafted_reply: str
    confidence: float
    retrieval_similarity: float
    escalation_flags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EscalationEngine:
    """Evaluates multi-factor criteria to output structured auto vs. escalate decisions."""

    RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "escalation_rules.yaml")

    def __init__(self, rules_path: Optional[str] = None):
        self.rules_path = rules_path or self.RULES_PATH
        self.rules = self._load_rules()
        self.criteria = self.rules.get("escalation_criteria", {})
        self.thresholds = self.criteria.get("thresholds", {})
        self.min_similarity = self.thresholds.get("min_retrieval_similarity", 0.15)
        self.min_intent_confidence = self.thresholds.get("min_intent_confidence", 0.60)

    def _load_rules(self) -> dict:
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def evaluate(
        self,
        customer_tweet: str,
        prediction: IntentPrediction,
        retrieved_pairs: List[RetrievedPair],
        drafted_reply: str,
    ) -> EscalationDecision:
        """Evaluate incoming message and state to decide auto vs escalate."""
        text_lower = customer_tweet.lower()
        flags: List[str] = []
        reasons: List[str] = []

        top_similarity = retrieved_pairs[0].score if retrieved_pairs else 0.0

        # Criterion 1: Hardware Hazard / Physical Safety (Immediate Escalate)
        hazard_cfg = self.criteria.get("hardware_hazard", {})
        for kw in hazard_cfg.get("keywords", []):
            if kw in text_lower:
                flags.append("HARDWARE_SAFETY_HAZARD")
                reasons.append(f"Reported physical safety hazard: '{kw}'")
                break

        # Criterion 2: Account Security Vulnerabilities (Immediate Escalate)
        sec_cfg = self.criteria.get("account_security", {})
        # Only escalate if security keywords exist or intent is account_security with sensitive keywords
        for kw in sec_cfg.get("keywords", []):
            if kw in text_lower:
                # Exclude benign mentions like "security questions setup" or general "security update"
                if "security update" in text_lower or "security patch" in text_lower:
                    continue
                flags.append("ACCOUNT_SECURITY_RISK")
                reasons.append(f"Security or account credentials concern detected: '{kw}'")
                break

        # Criterion 3: Severe Sentiment / Churn / Legal Risk (Immediate Escalate)
        sent_cfg = self.criteria.get("high_sentiment_risk", {})
        for kw in sent_cfg.get("keywords", []):
            if kw in text_lower:
                flags.append("HIGH_SENTIMENT_OR_LEGAL_RISK")
                reasons.append(f"High-friction customer sentiment / legal threat: '{kw}'")
                break

        # Criterion 4: Repeated Unresolved Troubleshooting (Escalate)
        rep_cfg = self.criteria.get("repeated_issue", {})
        for pat in rep_cfg.get("patterns", []):
            if pat in text_lower:
                flags.append("REPEATED_UNRESOLVED_ISSUE")
                reasons.append(f"Customer reported repeated failed attempts: '{pat}'")
                break

        # Criterion 5: Low Intent Confidence (Ambiguous query)
        if prediction.confidence < self.min_intent_confidence:
            flags.append("LOW_INTENT_CONFIDENCE")
            reasons.append(f"Ambiguous intent classification confidence ({prediction.confidence:.2f} < {self.min_intent_confidence})")

        # Criterion 6: Low Retrieval Confidence (Out of distribution)
        if top_similarity < self.min_similarity:
            flags.append("LOW_RETRIEVAL_SIMILARITY")
            reasons.append(f"Low historical retrieval match score ({top_similarity:.3f} < {self.min_similarity})")

        # Criterion 4b: Repair Dispute / Lost Package or Carrier Failure
        for term in ["lost my device", "lost the package", "no package on my porch", "unrepaired", "driver didn't knock"]:
            if term in text_lower:
                flags.append("ORDER_REPAIR_DISPUTE")
                reasons.append(f"Customer reported repair dispute or lost shipment: '{term}'")
                break

        # Criterion 7: Intent Default Escalations
        if prediction.intent == "complaint_frustration" and "HIGH_SENTIMENT_OR_LEGAL_RISK" not in flags:
            flags.append("INTENT_COMPLAINT_FRUSTRATION")
            reasons.append("Customer intent classified as escalated dissatisfaction / complaint.")
        elif prediction.intent == "account_security" and any(k in text_lower for k in ["locked", "lockout", "stolen", "compromised", "hacked", "recovery", "disabled", "fraud", "scam"]):
            if "ACCOUNT_SECURITY_RISK" not in flags:
                flags.append("ACCOUNT_SECURITY_RISK")
                reasons.append("Account security lockout or compromised credential requiring human verification.")

        # Determine Final Action
        if flags:
            action = "escalate"
            combined_reason = "; ".join(reasons)
        else:
            action = "auto"
            combined_reason = (
                f"Standard {prediction.intent} inquiry with confident intent "
                f"({prediction.confidence:.2f}) and verified historical grounding ({top_similarity:.2f})."
            )

        return EscalationDecision(
            intent=prediction.intent,
            action=action,
            reason=combined_reason,
            drafted_reply=drafted_reply,
            confidence=prediction.confidence,
            retrieval_similarity=top_similarity,
            escalation_flags=flags,
            metadata={
                "intent_method": prediction.method,
                "top_retrieved_thread": retrieved_pairs[0].thread_id if retrieved_pairs else None,
                "num_retrieved_pairs": len(retrieved_pairs),
            },
        )
