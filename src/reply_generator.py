"""Grounded Reply Generator for @AppleSupport persona.

Uses retrieved historical (customer query -> brand reply) pairs as in-context grounding
to draft brand-consistent, diagnostic, and helpful customer support responses.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.llm_client import LLMClient
from src.retrieval import RetrievedPair

logger = logging.getLogger(__name__)


class GroundedReplyGenerator:
    """Drafts customer support responses grounded in retrieved historical AppleSupport pairs."""

    SYSTEM_PROMPT = """You are the official Twitter customer support representative for @AppleSupport.
Your job is to draft an empathetic, concise, and helpful public Twitter reply to a customer inquiry.

Brand Guidelines for @AppleSupport:
1. Tone: Warm, polite, empathetic, and professional. Use phrases like "We're here to help", "We'd like to look into this with you", or "We understand how frustrating that can be".
2. Diagnostic First: Ask relevant diagnostic questions (e.g. Which iOS version? Which iPhone model? Does restarting help?).
3. Actionable Steps: Provide clear navigation paths (e.g., Settings > General > About, or Settings > Battery).
4. Direct Message (DM) Protocol: If the issue involves personal account information, serial numbers, billing details, or complex escalations, guide them to DM securely: "Please send us a DM so we can continue: https://twitter.com/messages/compose?recipient_id=AppleSupport"
5. Brevity: Keep the reply under 280 characters if possible, or up to two short paragraphs suitable for Twitter.
6. Grounding: Strictly incorporate relevant troubleshooting patterns, links, or diagnostic steps demonstrated in the provided historical examples. Do NOT hallucinate false policies.
"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def draft_reply(
        self,
        customer_tweet: str,
        intent: str,
        retrieved_pairs: List[RetrievedPair],
    ) -> str:
        """Draft a grounded response using retrieved historical context."""
        context_blocks = []
        for idx, pair in enumerate(retrieved_pairs, 1):
            context_blocks.append(
                f"--- Historical Example {idx} (Similarity: {pair.score}) ---\n"
                f"Customer: \"{pair.customer_text}\"\n"
                f"@AppleSupport Reply: \"{pair.brand_reply}\""
            )

        grounding_context = "\n\n".join(context_blocks) if context_blocks else "No historical matches found."

        prompt = (
            f"Classified Intent: {intent}\n\n"
            f"Relevant Historical Grounding Context:\n{grounding_context}\n\n"
            f"Incoming Customer Message:\n\"{customer_tweet}\"\n\n"
            "Draft a response from @AppleSupport following all brand guidelines."
        )

        try:
            reply = self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
            )
            cleaned_reply = reply.strip().strip('"')
            return cleaned_reply
        except Exception as e:
            logger.error(f"Error drafting reply ({e}). Falling back to template.")
            return self._fallback_reply(intent)

    def _fallback_reply(self, intent: str) -> str:
        """Fallback replies if generation fails."""
        templates = {
            "software_issue": "We'd like to help get this sorted out. Which device model and iOS version are you currently using? Does restarting your device temporarily help?",
            "hardware_battery": "We know how important your device is. Check Settings > Battery to see your battery usage, and let us know your device model so we can assist.",
            "account_security": "Your account security is our top priority. Please send us a direct message so we can securely look into this with you: https://twitter.com/messages/compose?recipient_id=AppleSupport",
            "billing_subscription": "We can help with your billing inquiry. You can review purchase history and request refunds at https://reportaproblem.apple.com, or send us a DM.",
            "how_to_inquiry": "We're happy to walk you through how to do that! Could you confirm which device model and iOS version you're working with?",
            "order_repair_status": "Thanks for checking in on your status. You can track your repair progress anytime at https://support.apple.com/repair, or DM us your Repair ID.",
            "complaint_frustration": "We sincerely apologize for your frustrating experience. Please send us a direct message with more details so a senior advisor can assist you.",
            "spam_or_irrelevant": "Thanks for reaching out! We're here if you ever need technical assistance with any Apple product.",
        }
        return templates.get(intent, "We're here to help! Please send us a DM with more details about what you're experiencing.")
