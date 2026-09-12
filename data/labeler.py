"""Interactive Terminal Tool for Rapid Ground Truth Annotation.

Allows human reviewers to review customer tweets, classify intents,
designate auto-handle vs. escalate decisions, and record gold reasoning notes.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional
import yaml

TAXONOMY_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "taxonomy.yaml")
GOLDEN_SET_FILE = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def load_taxonomy():
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("intents", {})


def display_guidelines(intents: Dict[str, dict]):
    print("\n" + "=" * 70)
    print("      APPLE SUPPORT GROUND TRUTH LABELING INTERFACE")
    print("=" * 70)
    print("Available Intents:")
    for idx, (key, meta) in enumerate(intents.items(), 1):
        print(f" [{idx}] {key:<22} -> {meta['name']}")
        print(f"     Definition: {meta['description']}")
    print("-" * 70)
    print("Escalation Guidelines:")
    print(" [1] auto     : Standard OS/hardware/how-to triage, link sharing, diagnostic questions.")
    print(" [2] escalate : Security breach, hacked Apple ID, battery hazard/swelling, severe anger, legal threat.")
    print("=" * 70 + "\n")


def label_unlabeled_file(input_file: str, output_file: str = GOLDEN_SET_FILE):
    intents = load_taxonomy()
    intent_keys = list(intents.keys())

    display_guidelines(intents)

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    items = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # Read existing labeled IDs
    labeled_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    labeled_ids.add(rec.get("id"))

    print(f"Loaded {len(items)} items. Already labeled: {len(labeled_ids)}.")
    
    labeled_count = 0
    with open(output_file, "a", encoding="utf-8") as out_f:
        for item in items:
            item_id = item.get("id") or item.get("tweet_id")
            if item_id in labeled_ids:
                continue

            text = item.get("customer_tweet") or item.get("customer_text") or item.get("text", "")
            print("\n" + "-" * 70)
            print(f"ID: {item_id}")
            print(f"Customer Tweet:\n\"{text}\"")
            print("-" * 70)

            # Intent selection
            while True:
                choice = input(f"Choose Intent (1-{len(intent_keys)}) or 'q' to quit: ").strip().lower()
                if choice == "q":
                    print(f"Session saved. Total newly labeled: {labeled_count}.")
                    return
                if choice.isdigit() and 1 <= int(choice) <= len(intent_keys):
                    chosen_intent = intent_keys[int(choice) - 1]
                    break
                elif choice in intent_keys:
                    chosen_intent = choice
                    break
                print("Invalid choice, try again.")

            # Action selection
            while True:
                act = input("Action: [1] auto  [2] escalate  (or 'q' to quit): ").strip().lower()
                if act == "q":
                    print(f"Session saved. Total newly labeled: {labeled_count}.")
                    return
                if act in ["1", "auto", "a"]:
                    chosen_action = "auto"
                    break
                elif act in ["2", "escalate", "e"]:
                    chosen_action = "escalate"
                    break
                print("Invalid choice. Enter 1 (auto) or 2 (escalate).")

            # Gold reasoning
            reason = input("Gold reasoning (press Enter for default): ").strip()
            if not reason:
                if chosen_action == "escalate":
                    reason = "Escalated to human tier due to policy triggers or complex multi-turn friction."
                else:
                    reason = "Standard diagnostic or how-to inquiry suitable for automated first response."

            record = {
                "id": str(item_id),
                "customer_tweet": text,
                "gold_intent": chosen_intent,
                "gold_action": chosen_action,
                "gold_reasoning": reason,
                "created_at": item.get("created_at"),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            labeled_ids.add(item_id)
            labeled_count += 1
            print(f"Saved [{item_id}].")

    print(f"\nAll done! Successfully labeled {labeled_count} items.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AppleSupport Interactive Labeler")
    parser.add_argument("--input", default=None, help="Path to raw or unlabeled JSONL file")
    parser.add_argument("--output", default=GOLDEN_SET_FILE, help="Path to output golden JSONL")
    args = parser.parse_args()

    if args.input:
        label_unlabeled_file(args.input, args.output)
    else:
        print("Usage: python labeler.py --input <unlabeled_tweets.jsonl>")
        intents = load_taxonomy()
        display_guidelines(intents)
