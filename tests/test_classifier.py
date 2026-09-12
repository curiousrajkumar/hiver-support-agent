"""Unit tests for IntentClassifier."""

import pytest
from src.intent_classifier import IntentClassifier


def test_taxonomy_loading():
    classifier = IntentClassifier()
    assert len(classifier.taxonomy) == 8
    assert "software_issue" in classifier.taxonomy
    assert "account_security" in classifier.taxonomy
    assert "hardware_battery" in classifier.taxonomy


def test_intent_classification_coverage():
    classifier = IntentClassifier()

    cases = [
        ("My iPhone screen is cracked and speaker is broken", "hardware_battery"),
        ("I was charged $9.99 for an in-app purchase, please refund", "billing_subscription"),
        ("Someone hacked into my Apple ID and changed my trusted phone number", "account_security"),
        ("How do I pair my AirPods to my iPad?", "how_to_inquiry"),
        ("Your customer service was horrible, taking this to the BBB and suing you", "complaint_frustration"),
        ("Good morning @AppleSupport have a nice day!", "spam_or_irrelevant"),
    ]

    for tweet, expected_intent in cases:
        pred = classifier.classify(tweet)
        assert pred.intent == expected_intent
        assert pred.confidence > 0.5
        assert pred.reasoning != ""
