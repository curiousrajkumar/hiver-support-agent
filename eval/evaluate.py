"""Comprehensive Evaluation Harness for AppleSupport Customer Agent vs. Baselines.

Evaluates Full System, Simple Baseline, and Trivial Baseline on 180 golden cases.
Computes:
1. Intent Classification: Accuracy, Macro F1, Weighted F1, Per-class metrics
2. Escalation Decision: Precision, Recall, F1, F2 (recall-weighted), Cost Asymmetry
3. Reply Quality (LLM-as-Judge): Faithfulness, Tone, Resolution, Brand Voice
Generates structured JSON benchmark results and prints headline comparison table.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

from eval.judge import ReplyQualityJudge
from src.agent import AppleSupportAgent
from src.baselines import SimpleBaselineAgent, TrivialBaselineAgent

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_set.jsonl")
RESULTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


def load_golden_set(filepath: str = GOLDEN_SET_PATH) -> List[dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Golden set not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_escalation_metrics(
    y_true: List[str],
    y_pred: List[str],
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> Dict[str, Any]:
    """Compute precision, recall, F1, F2, confusion matrix, and asymmetric cost."""
    # Positive class is "escalate"
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "escalate")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != "escalate" and yp == "escalate")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp != "escalate")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != "escalate" and yp != "escalate")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # F2 score puts 2x more weight on recall than precision: beta = 2
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0

    # Total cost penalty = (5 * FN) + (1 * FP)
    total_cost = (cost_fn * fn) + (cost_fp * fp)
    cost_per_query = round(total_cost / max(1, len(y_true)), 3)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "total_cost_penalty": total_cost,
        "cost_per_query": cost_per_query,
    }


def run_benchmark(sample_limit: Optional[int] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("      RUNNING BENCHMARK: FULL AGENT VS. SIMPLE & TRIVIAL BASELINES")
    print("=" * 80)

    dataset = load_golden_set()
    if sample_limit:
        dataset = dataset[:sample_limit]

    print(f"Loaded {len(dataset)} test instances from golden set.")

    # Initialize models
    trivial_agent = TrivialBaselineAgent(default_action="escalate")
    simple_agent = SimpleBaselineAgent()
    full_agent = AppleSupportAgent()
    judge = ReplyQualityJudge()

    models = {
        "trivial_baseline": trivial_agent,
        "simple_baseline": simple_agent,
        "full_system": full_agent,
    }

    y_true_intent = [item["gold_intent"] for item in dataset]
    y_true_action = [item["gold_action"] for item in dataset]

    benchmark_results: Dict[str, Any] = {}

    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}...")
        pred_intents = []
        pred_actions = []
        judge_scores = []

        # We evaluate judge on subset of 30 items to keep eval fast (<1 min)
        judge_sample_indices = set(range(0, len(dataset), max(1, len(dataset) // 30)))

        for idx, item in enumerate(dataset):
            tweet = item["customer_tweet"]
            decision = model.process(tweet)

            pred_intents.append(decision.intent)
            pred_actions.append(decision.action)

            # Judge scoring
            if idx in judge_sample_indices:
                score = judge.evaluate_reply(
                    customer_tweet=tweet,
                    drafted_reply=decision.drafted_reply,
                    retrieved_context=full_agent.retrieval.retrieve(tweet, top_k=2) if model_name == "full_system" else None,
                )
                judge_scores.append(score.to_dict())

        # 1. Intent Metrics
        acc = accuracy_score(y_true_intent, pred_intents)
        macro_f1 = f1_score(y_true_intent, pred_intents, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_intent, pred_intents, average="weighted", zero_division=0)

        # 2. Escalation Metrics
        esc_metrics = compute_escalation_metrics(y_true_action, pred_actions)

        # 3. Judge Metrics Average
        avg_faith = round(sum(s["grounding_faithfulness"] for s in judge_scores) / len(judge_scores), 2)
        avg_tone = round(sum(s["tone_match"] for s in judge_scores) / len(judge_scores), 2)
        avg_res = round(sum(s["resolves_the_issue"] for s in judge_scores) / len(judge_scores), 2)
        avg_voice = round(sum(s["brand_voice_consistency"] for s in judge_scores) / len(judge_scores), 2)
        avg_overall = round(sum(s["average_score"] for s in judge_scores) / len(judge_scores), 2)

        benchmark_results[model_name] = {
            "intent_metrics": {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "weighted_f1": round(float(weighted_f1), 4),
            },
            "escalation_metrics": esc_metrics,
            "judge_metrics": {
                "sample_size": len(judge_scores),
                "avg_grounding_faithfulness": avg_faith,
                "avg_tone_match": avg_tone,
                "avg_resolves_the_issue": avg_res,
                "avg_brand_voice_consistency": avg_voice,
                "overall_average": avg_overall,
            },
        }

    # Print Headline Comparison Table
    print("\n" + "=" * 88)
    print("                            HEADLINE RESULTS COMPARISON")
    print("=" * 88)
    header = f"{'Metric Area':<24} | {'Evaluation Metric':<26} | {'Trivial':<10} | {'Simple':<10} | {'Full System':<10}"
    print(header)
    print("-" * 88)

    rows = [
        ("Intent Classification", "Accuracy",
         f"{benchmark_results['trivial_baseline']['intent_metrics']['accuracy']*100:.1f}%",
         f"{benchmark_results['simple_baseline']['intent_metrics']['accuracy']*100:.1f}%",
         f"{benchmark_results['full_system']['intent_metrics']['accuracy']*100:.1f}%"),
        ("Intent Classification", "Macro F1",
         f"{benchmark_results['trivial_baseline']['intent_metrics']['macro_f1']:.3f}",
         f"{benchmark_results['simple_baseline']['intent_metrics']['macro_f1']:.3f}",
         f"{benchmark_results['full_system']['intent_metrics']['macro_f1']:.3f}"),
        ("Escalation Decision", "Precision (Escalate)",
         f"{benchmark_results['trivial_baseline']['escalation_metrics']['precision']:.3f}",
         f"{benchmark_results['simple_baseline']['escalation_metrics']['precision']:.3f}",
         f"{benchmark_results['full_system']['escalation_metrics']['precision']:.3f}"),
        ("Escalation Decision", "Recall (Escalate)",
         f"{benchmark_results['trivial_baseline']['escalation_metrics']['recall']:.3f}",
         f"{benchmark_results['simple_baseline']['escalation_metrics']['recall']:.3f}",
         f"{benchmark_results['full_system']['escalation_metrics']['recall']:.3f}"),
        ("Escalation Decision", "F2 Score (Recall-weighted)",
         f"{benchmark_results['trivial_baseline']['escalation_metrics']['f2']:.3f}",
         f"{benchmark_results['simple_baseline']['escalation_metrics']['f2']:.3f}",
         f"{benchmark_results['full_system']['escalation_metrics']['f2']:.3f}"),
        ("Escalation Decision", "Cost Penalty (5xFN + 1xFP)",
         f"{benchmark_results['trivial_baseline']['escalation_metrics']['total_cost_penalty']:.0f}",
         f"{benchmark_results['simple_baseline']['escalation_metrics']['total_cost_penalty']:.0f}",
         f"{benchmark_results['full_system']['escalation_metrics']['total_cost_penalty']:.0f}"),
        ("Reply Quality (Judge)", "Grounding Faithfulness",
         f"{benchmark_results['trivial_baseline']['judge_metrics']['avg_grounding_faithfulness']:.2f}/5",
         f"{benchmark_results['simple_baseline']['judge_metrics']['avg_grounding_faithfulness']:.2f}/5",
         f"{benchmark_results['full_system']['judge_metrics']['avg_grounding_faithfulness']:.2f}/5"),
        ("Reply Quality (Judge)", "Tone Match",
         f"{benchmark_results['trivial_baseline']['judge_metrics']['avg_tone_match']:.2f}/5",
         f"{benchmark_results['simple_baseline']['judge_metrics']['avg_tone_match']:.2f}/5",
         f"{benchmark_results['full_system']['judge_metrics']['avg_tone_match']:.2f}/5"),
        ("Reply Quality (Judge)", "Resolves the Issue",
         f"{benchmark_results['trivial_baseline']['judge_metrics']['avg_resolves_the_issue']:.2f}/5",
         f"{benchmark_results['simple_baseline']['judge_metrics']['avg_resolves_the_issue']:.2f}/5",
         f"{benchmark_results['full_system']['judge_metrics']['avg_resolves_the_issue']:.2f}/5"),
        ("Reply Quality (Judge)", "Brand Voice Consistency",
         f"{benchmark_results['trivial_baseline']['judge_metrics']['avg_brand_voice_consistency']:.2f}/5",
         f"{benchmark_results['simple_baseline']['judge_metrics']['avg_brand_voice_consistency']:.2f}/5",
         f"{benchmark_results['full_system']['judge_metrics']['avg_brand_voice_consistency']:.2f}/5"),
    ]

    for cat, name, triv, simp, full in rows:
        print(f"{cat:<24} | {name:<26} | {triv:<10} | {simp:<10} | {full:<10}")

    print("=" * 88)

    # Save to JSON
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nFull benchmark results saved to: {RESULTS_OUTPUT_PATH}\n")

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()
