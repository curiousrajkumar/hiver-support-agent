"""Judge-vs-Human Agreement & Correlation Analysis.

Computes Spearman rank correlation and Cohen's Kappa between LLM-as-judge scores
and human ground truth scores across 30 representative AppleSupport interactions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from eval.judge import ReplyQualityJudge
from src.agent import AppleSupportAgent

logger = logging.getLogger(__name__)

HUMAN_SUBSET_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "human_eval_subset.jsonl")
REPORT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "human_agreement_report.json")


def run_human_agreement_analysis(subset_path: str = HUMAN_SUBSET_FILE) -> Dict[str, object]:
    print("=" * 70)
    print("      LLM-AS-JUDGE VS. HUMAN ANNOTATOR AGREEMENT ANALYSIS")
    print("=" * 70)

    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Human evaluation subset file not found: {subset_path}")

    items = []
    with open(subset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} human-annotated samples.")

    agent = AppleSupportAgent()
    judge = ReplyQualityJudge()

    human_faith, judge_faith = [], []
    human_tone, judge_tone = [], []
    human_res, judge_res = [], []
    human_voice, judge_voice = [], []
    human_overall, judge_overall = [], []

    detailed_records = []

    for idx, item in enumerate(items, 1):
        tweet = item.get("tweet", "")
        # Run agent
        decision = agent.process(tweet)
        retrieved = agent.retrieval.retrieve(tweet, top_k=3)

        # Run judge
        score = judge.evaluate_reply(
            customer_tweet=tweet,
            drafted_reply=decision.drafted_reply,
            retrieved_context=retrieved,
        )

        h_f = item["human_faithfulness"]
        h_t = item["human_tone"]
        h_r = item["human_resolution"]
        h_v = item["human_brand_voice"]
        h_avg = round((h_f + h_t + h_r + h_v) / 4.0, 2)

        j_f = score.grounding_faithfulness
        j_t = score.tone_match
        j_r = score.resolves_the_issue
        j_v = score.brand_voice_consistency
        j_avg = score.average_score

        human_faith.append(h_f)
        judge_faith.append(j_f)

        human_tone.append(h_t)
        judge_tone.append(j_t)

        human_res.append(h_r)
        judge_res.append(j_r)

        human_voice.append(h_v)
        judge_voice.append(j_v)

        human_overall.append(h_avg)
        judge_overall.append(j_avg)

        detailed_records.append({
            "id": item["id"],
            "tweet": tweet,
            "drafted_reply": decision.drafted_reply,
            "human_scores": {"faithfulness": h_f, "tone": h_t, "resolution": h_r, "brand_voice": h_v, "average": h_avg},
            "judge_scores": {"faithfulness": j_f, "tone": j_t, "resolution": j_r, "brand_voice": j_v, "average": j_avg},
        })

    def calc_stats(h_list: List[int], j_list: List[int]) -> Dict[str, float]:
        rho, p_val = spearmanr(h_list, j_list)
        # Handle nan if constant arrays
        if np.isnan(rho):
            rho = 1.0 if h_list == j_list else 0.0
            p_val = 0.0

        # Cohen's kappa (quadratic weighted for ordinal rating scales)
        try:
            kappa = cohen_kappa_score(h_list, j_list, weights="quadratic")
        except Exception:
            kappa = cohen_kappa_score(h_list, j_list)

        if np.isnan(kappa):
            kappa = 1.0 if h_list == j_list else 0.0

        mae = float(np.mean(np.abs(np.array(h_list) - np.array(j_list))))
        exact_match = float(np.mean(np.array(h_list) == np.array(j_list)) * 100.0)

        return {
            "spearman_rho": round(float(rho), 3),
            "spearman_p_value": round(float(p_val), 4),
            "cohens_kappa_quadratic": round(float(kappa), 3),
            "mean_absolute_error": round(mae, 3),
            "exact_agreement_pct": round(exact_match, 1),
        }

    report = {
        "sample_size": len(items),
        "metrics": {
            "grounding_faithfulness": calc_stats(human_faith, judge_faith),
            "tone_match": calc_stats(human_tone, judge_tone),
            "resolves_the_issue": calc_stats(human_res, judge_res),
            "brand_voice_consistency": calc_stats(human_voice, judge_voice),
            "overall_average": {
                "spearman_rho": round(float(spearmanr(human_overall, judge_overall)[0]), 3),
                "mean_absolute_error": round(float(np.mean(np.abs(np.array(human_overall) - np.array(judge_overall)))), 3),
            },
        },
        "sample_evaluations": detailed_records[:5],
    }

    print("\n--- RESULTS TABLE ---")
    print(f"{'Dimension':<28} | {'Spearman Rho':<14} | {'Quadratic Kappa':<16} | {'MAE':<8} | {'Exact Match %'}")
    print("-" * 80)
    for dim in ["grounding_faithfulness", "tone_match", "resolves_the_issue", "brand_voice_consistency"]:
        m = report["metrics"][dim]
        print(f"{dim:<28} | {m['spearman_rho']:<14} | {m['cohens_kappa_quadratic']:<16} | {m['mean_absolute_error']:<8} | {m['exact_agreement_pct']}%")

    print("-" * 80)
    overall_rho = report["metrics"]["overall_average"]["spearman_rho"]
    overall_mae = report["metrics"]["overall_average"]["mean_absolute_error"]
    print(f"{'OVERALL AGGREGATE':<28} | {overall_rho:<14} | {'--':<16} | {overall_mae:<8} | --")
    print("=" * 80)

    # Save to file
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {REPORT_OUTPUT_FILE}\n")

    return report


if __name__ == "__main__":
    run_human_agreement_analysis()
