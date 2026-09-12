"""Single-command runner for the AppleSupport AI Customer Support Agent pipeline.

Usage:
  python run_pipeline.py               # Interactive demo on representative customer queries
  python run_pipeline.py --mode eval   # Run full benchmark evaluation vs baselines + human agreement
  python run_pipeline.py --query "My battery drains 20% in 15 mins after iOS 11 update"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.evaluate import run_benchmark
from eval.human_agreement import run_human_agreement_analysis
from src.agent import AppleSupportAgent


def run_interactive_demo():
    print("=" * 80)
    print("      @APPLESUPPORT AI CUSTOMER SUPPORT AGENT - LIVE DEMO")
    print("=" * 80)
    print("Initializing pipeline (Intent Classifier -> BM25 Grounding -> Reply Drafter -> Escalation)...")
    start_time = time.time()

    agent = AppleSupportAgent()
    init_duration = time.time() - start_time
    print(f"Pipeline ready in {init_duration:.2f}s.\n")

    test_queries = [
        ("Software Glitch", "Ever since updating to iOS 11.1 my battery drops 20% in 15 minutes and apps keep stuttering."),
        ("Physical Safety Hazard", "My screen is bulging out and lifting off the frame near the volume buttons, is the battery swelling?"),
        ("Account Security Alert", "Keep getting verification codes sent to my phone that I didn't request! Has someone hacked my account?"),
        ("Billing / Refund", "I was charged $9.99 for an app subscription I cancelled two weeks ago, need a refund."),
        ("Feature How-To", "How do I transfer photos from my old iPhone to my Mac without using iCloud?"),
        ("Severe Complaint / Legal Risk", "Fourth time my MacBook has kernel panicked. Your Genius Bar tech was completely dismissive. See you in court!"),
        ("Social Chatter / Spam", "Hey @AppleSupport hope you have a nice Friday! What is Tim Cook's favorite color?"),
    ]

    for category, query in test_queries:
        print("-" * 80)
        print(f"Scenario: [{category}]")
        print(f"Customer Tweet:\n  \"{query}\"")
        
        t0 = time.time()
        decision = agent.process(query)
        latency = (time.time() - t0) * 1000

        action_badge = "[!] ESCALATE TO HUMAN" if decision.action == "escalate" else "[OK] AUTO-HANDLE"
        print(f"\nDecision: {action_badge} (Latency: {latency:.1f}ms)")
        print(f"  * Classified Intent: {decision.intent} (Confidence: {decision.confidence:.2f})")
        print(f"  * Retrieval Grounding Match: {decision.retrieval_similarity:.3f}")
        print(f"  * Policy Reasoning: {decision.reason}")
        if decision.escalation_flags:
            print(f"  * Active Escalation Flags: {', '.join(decision.escalation_flags)}")
        print(f"\nDrafted Public @AppleSupport Reply:\n  \"{decision.drafted_reply}\"")
        print("-" * 80 + "\n")

    print("=" * 80)
    print("Demo complete! To run the full quantitative evaluation suite against all baselines, run:")
    print("  python run_pipeline.py --mode eval")
    print("=" * 80)


def run_custom_query(query: str):
    agent = AppleSupportAgent()
    decision = agent.process(query)
    print("\n" + "=" * 60)
    print("CUSTOMER SUPPORT DECISION OBJECT")
    print("=" * 60)
    print(json.dumps(decision.to_dict(), indent=2))
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AppleSupport AI Support Pipeline Runner")
    parser.add_argument(
        "--mode",
        choices=["demo", "eval", "agreement"],
        default="demo",
        help="Execution mode: demo (sample scenarios), eval (full benchmark vs baselines), agreement (judge vs human correlation)",
    )
    parser.add_argument("--query", type=str, default=None, help="Process a single custom tweet message")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of retrieval index from CSV")

    args = parser.parse_args()

    if args.rebuild_index:
        agent = AppleSupportAgent()
        agent.retrieval.initialize(force_rebuild=True)
        print("Retrieval index rebuilt successfully.")

    if args.query:
        run_custom_query(args.query)
    elif args.mode == "demo":
        run_interactive_demo()
    elif args.mode == "eval":
        run_benchmark()
        run_human_agreement_analysis()
    elif args.mode == "agreement":
        run_human_agreement_analysis()


if __name__ == "__main__":
    main()
