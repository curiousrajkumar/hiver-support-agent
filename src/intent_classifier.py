"""Intent Classifier for AppleSupport customer queries.

Combines taxonomy definitions, few-shot prompt construction, and strict JSON output validation.
Includes regex/keyword baseline fallback for offline execution and ablation testing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    reasoning: str
    method: str  # "llm_few_shot" | "regex_baseline"


class IntentClassifier:
    """Classifies customer tweets into one of 8 AppleSupport intents."""

    TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "taxonomy.yaml")

    def __init__(
        self,
        taxonomy_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        default_method: str = "llm_few_shot",
    ):
        self.taxonomy_path = taxonomy_path or self.TAXONOMY_PATH
        self.taxonomy = self._load_taxonomy()
        self.llm = llm_client or LLMClient()
        self.default_method = default_method
        self.few_shot_prompt_template = self._build_system_prompt()

    def _load_taxonomy(self) -> Dict[str, dict]:
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("intents", {})

    def _build_system_prompt(self) -> str:
        lines = [
            "You are an expert customer intent classification model for @AppleSupport on Twitter.",
            "Analyze the incoming customer tweet and classify it into EXACTLY ONE of the following 8 intents:",
            "",
        ]
        for key, info in self.taxonomy.items():
            lines.append(f"Intent Key: {key}")
            lines.append(f"Name: {info['name']}")
            lines.append(f"Definition: {info['description']}")
            lines.append("Examples:")
            for ex in info.get("examples", []):
                lines.append(f"  - \"{ex}\"")
            lines.append("")

        lines.extend([
            "Instructions:",
            "1. Output your answer strictly as a valid JSON object with keys: 'intent', 'confidence', 'reasoning'.",
            "2. 'intent' MUST be one of: " + ", ".join(list(self.taxonomy.keys())),
            "3. 'confidence' must be a float between 0.0 and 1.0.",
            "4. 'reasoning' must be a single concise sentence justifying the classification.",
            "Do NOT include markdown formatting or extra commentary outside the JSON object.",
        ])
        return "\n".join(lines)

    def classify(self, text: str, method: Optional[str] = None) -> IntentPrediction:
        """Classify customer message using specified method."""
        chosen_method = method or self.default_method

        if chosen_method == "regex_baseline":
            return self._classify_regex(text)

        # Default: Few-shot LLM
        prompt = (
            f"Customer Tweet: \"{text}\"\n\n"
            "Classify the intent into valid JSON with keys: intent, confidence, reasoning."
        )

        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.few_shot_prompt_template,
                response_format="json",
            )
            parsed = self._extract_json(raw_response)
            intent = parsed.get("intent", "").strip().lower()

            if intent in self.taxonomy:
                return IntentPrediction(
                    intent=intent,
                    confidence=float(parsed.get("confidence", 0.90)),
                    reasoning=parsed.get("reasoning", "Classified via LLM few-shot reasoning."),
                    method="llm_few_shot",
                )
            else:
                logger.warning(f"LLM produced unknown intent '{intent}'. Falling back to regex.")
                return self._classify_regex(text)

        except Exception as e:
            logger.error(f"Error in LLM classification ({e}). Falling back to regex.")
            return self._classify_regex(text)

    def _extract_json(self, raw_text: str) -> dict:
        """Extract JSON from possible markdown wrappers."""
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        return json.loads(text)

    def _classify_regex(self, text: str) -> IntentPrediction:
        """Lightweight regex/keyword classifier baseline."""
        lower = text.lower()

        # 1. Spam or Irrelevant
        if any(w in lower for w in ["follow for follow", "rt to win", "mixtape", "soundcloud", "clowns", "good morning", "happy friday"]):
            return IntentPrediction(
                intent="spam_or_irrelevant",
                confidence=0.85,
                reasoning="Matched social greeting or promotional spam patterns.",
                method="regex_baseline",
            )

        # 2. Account Security
        if any(w in lower for w in ["hacked", "stolen", "activation lock", "apple id", "locked out", "2fa", "two-factor", "verification code", "compromised"]):
            return IntentPrediction(
                intent="account_security",
                confidence=0.90,
                reasoning="Matched account security and credentials keywords.",
                method="regex_baseline",
            )

        # 3. Complaint & Frustration
        if any(w in lower for w in ["lawyer", "sue", "terrible", "worst", "disgrace", "horrible", "unacceptable", "done with apple", "furious"]):
            return IntentPrediction(
                intent="complaint_frustration",
                confidence=0.92,
                reasoning="Matched severe customer dissatisfaction and threat keywords.",
                method="regex_baseline",
            )

        # 4. Hardware & Battery
        if any(w in lower for w in ["battery", "drain", "shuts down", "screen cracked", "lifting", "swelling", "speaker", "earpiece", "home button", "charger cable"]):
            return IntentPrediction(
                intent="hardware_battery",
                confidence=0.88,
                reasoning="Matched hardware component and battery degradation terms.",
                method="regex_baseline",
            )

        # 5. Billing & Subscription
        if any(w in lower for w in ["refund", "charge", "charged", "subscription", "declined", "app store purchase", "billed"]):
            return IntentPrediction(
                intent="billing_subscription",
                confidence=0.89,
                reasoning="Matched billing and subscription keywords.",
                method="regex_baseline",
            )

        # 6. Order & Repair Status
        if any(w in lower for w in ["repair", "repair id", "tracking", "status", "shipment", "preparing to ship", "genius bar appointment", "trade-in"]):
            return IntentPrediction(
                intent="order_repair_status",
                confidence=0.87,
                reasoning="Matched order tracking and Genius Bar repair status terms.",
                method="regex_baseline",
            )

        # 7. How-To Inquiry
        if any(w in lower for w in ["how do i", "how to", "how can i", "pair", "transfer", "backup", "turn off read receipts", "customize"]):
            return IntentPrediction(
                intent="how_to_inquiry",
                confidence=0.86,
                reasoning="Matched instructional how-to query structure.",
                method="regex_baseline",
            )

        # 8. Default: Software Issue
        return IntentPrediction(
            intent="software_issue",
            confidence=0.75,
            reasoning="Default classification for technical glitch/bug inquiry.",
            method="regex_baseline",
        )
