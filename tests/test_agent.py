"""Integration tests for end-to-end AppleSupportAgent and baselines."""

import pytest
from src.agent import AppleSupportAgent
from src.baselines import SimpleBaselineAgent, TrivialBaselineAgent


def test_agent_end_to_end_decision_schema():
    agent = AppleSupportAgent()
    decision = agent.process("How do I back up my iPhone to my Mac?")

    assert decision.intent in ["how_to_inquiry", "software_issue"]
    assert decision.action in ["auto", "escalate"]
    assert decision.drafted_reply != ""
    assert decision.confidence > 0.0
    assert isinstance(decision.escalation_flags, list)
    assert "num_retrieved_pairs" in decision.metadata


def test_trivial_baseline():
    baseline = TrivialBaselineAgent()
    decision = baseline.process("Any message here")
    assert decision.action == "escalate"
    assert "canned" in decision.reason.lower() or "trivial" in decision.reason.lower()


def test_simple_baseline():
    baseline = SimpleBaselineAgent()
    decision_normal = baseline.process("How do I change my volume?")
    assert decision_normal.action == "auto"

    decision_threat = baseline.process("I am going to sue you and hire a lawyer!")
    assert decision_threat.action == "escalate"
