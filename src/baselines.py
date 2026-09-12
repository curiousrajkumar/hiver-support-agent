"""Baseline Models for AppleSupport Customer Agent Benchmarking.

1. Trivial Baseline: Always majority intent, always canned reply, always escalate (or always auto).
2. Simple Baseline: Regex keyword intent classifier, static template reply (no retrieval grounding), simple keyword escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.escalation import EscalationDecision
from src.intent_classifier import IntentClassifier


class TrivialBaselineAgent:
    """Trivial baseline: Static canned response and constant escalation action."""

    CANNED_REPLY = (
        "Thanks for reaching out to @AppleSupport! We're here to help. "
        "Please send us a direct message so we can look into this with you: "
        "https://twitter.com/messages/compose?recipient_id=AppleSupport"
    )

    def __init__(self, default_action: str = "escalate"):
        self.default_action = default_action

    def process(self, customer_tweet: str) -> EscalationDecision:
        return EscalationDecision(
            intent="software_issue",  # Majority class
            action=self.default_action,
            reason=f"Trivial baseline default decision ({self.default_action}).",
            drafted_reply=self.CANNED_REPLY,
            confidence=0.50,
            retrieval_similarity=0.0,
            escalation_flags=["TRIVIAL_BASELINE_DEFAULT"] if self.default_action == "escalate" else [],
            metadata={"model": "trivial_baseline"},
        )


class SimpleBaselineAgent:
    """Simple baseline: Keyword/regex intent classifier + static template replies + basic keyword escalation."""

    TEMPLATES = {
        "software_issue": "We understand your device is experiencing issues. Please try restarting your device to see if that helps.",
        "hardware_battery": "Battery and hardware performance is important. Please check Settings > Battery for details.",
        "account_security": "If you are having trouble with your Apple ID, please visit https://iforgot.apple.com for assistance.",
        "billing_subscription": "For billing issues or subscription cancellations, please visit https://reportaproblem.apple.com.",
        "how_to_inquiry": "You can find step-by-step instructions for settings and features on https://support.apple.com.",
        "order_repair_status": "You can check the status of your order or repair anytime at https://support.apple.com/repair.",
        "complaint_frustration": "We apologize for the inconvenience. Please contact our support team via DM.",
        "spam_or_irrelevant": "Thank you for reaching out to Apple Support.",
    }

    ESCALATION_KEYWORDS = ["hacked", "stolen", "sue", "lawyer", "refund", "horrible", "terrible", "swelling", "smoke"]

    def __init__(self):
        self.classifier = IntentClassifier()

    def process(self, customer_tweet: str) -> EscalationDecision:
        # 1. Regex intent classification
        prediction = self.classifier.classify(customer_tweet, method="regex_baseline")

        # 2. Basic keyword escalation
        text_lower = customer_tweet.lower()
        escalate = any(kw in text_lower for kw in self.ESCALATION_KEYWORDS)
        action = "escalate" if escalate else "auto"
        reason = "Matched basic heuristic blacklist keyword." if escalate else "No blacklist keywords found."

        # 3. Static ungrounded template reply
        reply = self.TEMPLATES.get(prediction.intent, self.TEMPLATES["software_issue"])

        return EscalationDecision(
            intent=prediction.intent,
            action=action,
            reason=reason,
            drafted_reply=reply,
            confidence=prediction.confidence,
            retrieval_similarity=0.0,  # No retrieval grounding
            escalation_flags=["SIMPLE_KEYWORD_MATCH"] if escalate else [],
            metadata={"model": "simple_baseline", "intent_method": "regex"},
        )
