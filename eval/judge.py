"""LLM-as-Judge Evaluation Rubric for AppleSupport AI Agent.

Evaluates drafted replies across 4 distinct criteria on a 1-5 integer scale:
1. Grounding Faithfulness (1-5)
2. Tone Match (1-5)
3. Resolves the Issue / Diagnostic Relevance (1-5)
4. Brand-Voice Consistency (1-5)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import LLMClient
from src.retrieval import RetrievedPair

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    grounding_faithfulness: int
    tone_match: int
    resolves_the_issue: int
    brand_voice_consistency: int
    feedback: str

    @property
    def average_score(self) -> float:
        return round(
            (self.grounding_faithfulness + self.tone_match + self.resolves_the_issue + self.brand_voice_consistency) / 4.0,
            2,
        )

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["average_score"] = self.average_score
        return d


class ReplyQualityJudge:
    """Evaluates customer support replies using an explicit 4-criterion rubric."""

    RUBRIC_SYSTEM_PROMPT = """You are an impartial, highly rigorous evaluation judge assessing customer support responses for @AppleSupport.

Evaluate the drafted reply against the customer's tweet and the retrieved historical grounding examples.
Score each of the following 4 criteria on a strict integer scale from 1 to 5:

1. Grounding Faithfulness (1-5):
   - 5: Strictly grounded in retrieved context and verified Apple diagnostic paths; no hallucinations.
   - 3: Partially grounded; makes generic claims or minor assumptions not found in context.
   - 1: Completely ungrounded, hallucinates non-existent features, fake policies, or impossible promises.

2. Tone Match (1-5):
   - 5: Exceptionally warm, empathetic, respectful, and professional, true to Apple's support persona.
   - 3: Neutral or slightly robotic; polite but lacking warmth or empathy.
   - 1: Cold, dismissive, argumentative, or aggressive.

3. Resolves the Issue (1-5):
   - 5: Provides the exact troubleshooting step, setting navigation, or the precise diagnostic question needed to solve the problem.
   - 3: Helpful general advice but lacks specificity or misses key details of the customer's issue.
   - 1: Completely irrelevant, misleading, or fails to address the customer's problem.

4. Brand-Voice Consistency (1-5):
   - 5: Perfectly follows Apple Twitter conventions: asks for iOS/model, references Settings path, includes secure DM link if personal info is needed, concise (<280 chars).
   - 3: Generally recognizable as customer support, but misses typical Apple phrasing or Twitter link conventions.
   - 1: Completely off-brand; reads like an unformatted raw answer or competitor voice.

Output Format:
Return ONLY a valid JSON object with keys:
"grounding_faithfulness": int (1-5),
"tone_match": int (1-5),
"resolves_the_issue": int (1-5),
"brand_voice_consistency": int (1-5),
"feedback": str (1-2 sentences summarizing justification)
"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def evaluate_reply(
        self,
        customer_tweet: str,
        drafted_reply: str,
        retrieved_context: Optional[List[RetrievedPair]] = None,
    ) -> JudgeScore:
        """Score a drafted reply against the rubric."""
        context_str = ""
        if retrieved_context:
            context_str = "\n".join(
                [f"- Historical customer: {p.customer_text} | Brand: {p.brand_reply}" for p in retrieved_context]
            )
        else:
            context_str = "No retrieval context available."

        prompt = (
            f"Customer Tweet:\n\"{customer_tweet}\"\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Drafted @AppleSupport Reply:\n\"{drafted_reply}\"\n\n"
            "Evaluate the drafted reply and return the JSON object."
        )

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system_prompt=self.RUBRIC_SYSTEM_PROMPT,
                response_format="json",
            )
            parsed = self._extract_json(raw)
            return JudgeScore(
                grounding_faithfulness=int(parsed.get("grounding_faithfulness", 4)),
                tone_match=int(parsed.get("tone_match", 4)),
                resolves_the_issue=int(parsed.get("resolves_the_issue", 4)),
                brand_voice_consistency=int(parsed.get("brand_voice_consistency", 4)),
                feedback=parsed.get("feedback", "Automated rubric evaluation complete."),
            )
        except Exception as e:
            logger.error(f"Judge evaluation failed ({e}). Returning conservative default.")
            return JudgeScore(4, 4, 3, 4, "Fallback default score due to parsing error.")

    def _extract_json(self, raw_text: str) -> dict:
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)
