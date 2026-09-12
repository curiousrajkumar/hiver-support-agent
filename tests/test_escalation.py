"""Unit tests for EscalationEngine."""

import pytest
from src.escalation import EscalationEngine
from src.intent_classifier import IntentPrediction
from src.retrieval import RetrievedPair


def test_safety_hazard_triggers_escalation():
    engine = EscalationEngine()
    prediction = IntentPrediction(intent="hardware_battery", confidence=0.9, reasoning="Hardware", method="llm")
    retrieved = [RetrievedPair("t1", "cust", "reply", 0.5)]

    decision = engine.evaluate(
        customer_tweet="My screen is lifting off and the battery is swelling up",
        prediction=prediction,
        retrieved_pairs=retrieved,
        drafted_reply="Reply",
    )

    assert decision.action == "escalate"
    assert "HARDWARE_SAFETY_HAZARD" in decision.escalation_flags


def test_account_security_triggers_escalation():
    engine = EscalationEngine()
    prediction = IntentPrediction(intent="account_security", confidence=0.95, reasoning="Security", method="llm")
    retrieved = [RetrievedPair("t1", "cust", "reply", 0.4)]

    decision = engine.evaluate(
        customer_tweet="Help someone hacked my account and made unauthorized purchases!",
        prediction=prediction,
        retrieved_pairs=retrieved,
        drafted_reply="Reply",
    )

    assert decision.action == "escalate"
    assert "ACCOUNT_SECURITY_RISK" in decision.escalation_flags


def test_low_retrieval_confidence_triggers_escalation():
    engine = EscalationEngine()
    prediction = IntentPrediction(intent="software_issue", confidence=0.85, reasoning="Software", method="llm")
    # Score 0.05 is below 0.15 threshold
    retrieved = [RetrievedPair("t1", "cust", "reply", 0.05)]

    decision = engine.evaluate(
        customer_tweet="Unseen exotic error code 0x99A8BC1 on third-party peripheral",
        prediction=prediction,
        retrieved_pairs=retrieved,
        drafted_reply="Reply",
    )

    assert decision.action == "escalate"
    assert "LOW_RETRIEVAL_SIMILARITY" in decision.escalation_flags


def test_standard_query_auto_handles():
    engine = EscalationEngine()
    prediction = IntentPrediction(intent="how_to_inquiry", confidence=0.92, reasoning="How to", method="llm")
    retrieved = [RetrievedPair("t1", "cust", "reply", 0.65)]

    decision = engine.evaluate(
        customer_tweet="How do I turn off read receipts for one person in iMessage?",
        prediction=prediction,
        retrieved_pairs=retrieved,
        drafted_reply="Reply",
    )

    assert decision.action == "auto"
    assert len(decision.escalation_flags) == 0
