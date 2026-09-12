# AppleSupport AI Customer Support Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-green.svg)](https://pytest.org)

An end-to-end AI Customer Support pipeline engineered for **@AppleSupport** based on the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`).

The system reconstructs customer-brand conversation threads, classifies customer intents into an 8-category taxonomy, grounds responses in retrieved historical AppleSupport pairs via BM25 Okapi, executes multi-factor auto-handle vs. escalate decisions under asymmetric operational costs, and evaluates response quality using an LLM-as-judge rubric correlated with human ground truth.

---

## System Architecture

```
                    INCOMING CUSTOMER TWEET
                              │
       ┌──────────────────────┴──────────────────────┐
       │                                             │
       ▼                                             ▼
[Intent Classifier]                       [Grounded Retrieval]
• 8-class taxonomy                        • BM25 Okapi search
• Few-shot prompt                         • Pre-indexed historical pairs
• Confidence scoring                      • Normalised similarity score
       │                                             │
       └──────────────────────┬──────────────────────┘
                              ▼
                   [Multi-Factor Escalation Engine]
                   • Account Security (hacked, 2FA, stolen)
                   • Physical Hazards (battery swelling, smoke)
                   • Hostile Churn & Legal (lawsuits, threats)
                   • Low Retrieval Match (< 0.15 similarity)
                   • Ambiguous Intent (< 0.60 confidence)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       [Action: ESCALATE]              [Action: AUTO-HANDLE]
       • Routes to Human Queue         • Drafts public @AppleSupport reply
       • Flag: Policy Reason           • Grounded in retrieved pairs
       • DM Escalation Link            • Empathy + Diagnostic Questions
```

---

## Step-by-Step Guide: How to Run This Project

The repository is completely self-contained and pre-indexed. You can reproduce the full system and benchmark results in **under 2 minutes** with zero external API keys or configuration.

---

### Step 1: Open Terminal & Navigate to Project
Open your terminal (PowerShell, Command Prompt, or bash) and enter the project directory:
```bash
cd d:\hiver-support-agent
```

---

### Step 2: (Optional) Set Up Virtual Environment
Creating a clean virtual environment is recommended:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Or Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Or macOS / Linux:
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies
Install the required packages (`pyyaml`, `scikit-learn`, `pandas`, `rank-bm25`, `pytest`):
```bash
pip install -r requirements.txt
```
*(All dependencies install in under 60 seconds with no GPU or external build tools required).*

---

### Step 4: Run the Interactive Live Demo
Test the pipeline across 7 distinct real-world customer support scenarios (physical battery hazard, hacked account, billing refund, iOS software glitch, feature how-to, severe legal complaint, social spam):
```bash
python run_pipeline.py
# Or using Makefile:
make run
```
**What to expect:**
* Initializes in **0.01 seconds**.
* Displays the customer query, classified intent, confidence score, retrieval grounding similarity, active policy escalation flags, policy reasoning, and the drafted public `@AppleSupport` reply.
* Output shows `[OK] AUTO-HANDLE` for standard queries and `[!] ESCALATE TO HUMAN` for safety/security/legal threats.

---

### Step 5: Test with Custom Tweets / Inquiries
Pass any arbitrary customer tweet using the `--query` flag to inspect the structured decision object:
```bash
# Example A: Physical safety hazard (battery swelling)
python run_pipeline.py --query "My screen is bulging out and getting very hot"

# Example B: Account security breach (2FA compromise)
python run_pipeline.py --query "Someone hacked my Apple ID and changed my trusted phone number!"

# Example C: Standard feature how-to
python run_pipeline.py --query "How do I transfer photos from my old iPhone to my Mac without using iCloud?"
```
**What to expect:** Returns a clean, structured JSON decision object:
```json
{
  "intent": "hardware_battery",
  "action": "escalate",
  "reason": "Reported physical safety hazard: 'bulging'",
  "drafted_reply": "We know how important your hardware and battery performance are...",
  "confidence": 0.92,
  "retrieval_similarity": 0.428,
  "escalation_flags": ["HARDWARE_SAFETY_HAZARD"]
}
```

---

### Step 6: Run Full Quantitative Benchmark vs. Baselines
Evaluate the **Full System**, **Simple Baseline** (regex + templates), and **Trivial Baseline** (always escalate) across the 180 golden test instances, plus compute judge-vs-human agreement on 30 items:
```bash
python run_pipeline.py --mode eval
# Or using Makefile:
make eval
```
**What to expect:**
* Runs in ~5 seconds.
* Prints the comprehensive Headline Comparison Table comparing Accuracy, Macro F1, Precision, Recall, $F_2$, Cost Penalty ($5 \times FN + 1 \times FP$), and LLM Judge scores.
* Prints the Human-Agreement correlation table (Spearman $\rho$ and Quadratic Weighted Cohen's $\kappa$).
* Saves full machine-readable results to `eval/benchmark_results.json` and `eval/human_agreement_report.json`.

---

### Step 7: Run Automated Test Suite
Execute the unit and integration tests covering data loading, thread reconstruction, retrieval indexing, intent classification, and multi-factor escalation:
```bash
pytest -v
# Or using Makefile:
make test
```
**What to expect:** All **14 tests pass in ~0.25 seconds**.

---

### Step 8: (Optional) Interactive Ground-Truth Labeling Tool
To inspect, review, or label customer tweets with built-in guidelines:
```bash
python data/labeler.py
# Or using Makefile:
make label
```

---

### Step 9: (Optional) Use Live LLM APIs (OpenAI / Anthropic / Gemini)
By default, the pipeline uses a high-fidelity offline simulator for deterministic evaluation without paid API keys. If you want to use live LLMs:
```bash
# For OpenAI:
set OPENAI_API_KEY="your_key_here"       # Windows CMD
$env:OPENAI_API_KEY="your_key_here"      # Windows PowerShell
export OPENAI_API_KEY="your_key_here"    # macOS / Linux

# Or for Anthropic:
$env:ANTHROPIC_API_KEY="your_key_here"

# Or for Google Gemini:
$env:GEMINI_API_KEY="your_key_here"
```
The client automatically detects your active key and switches to live generation.

---

### Step 10: (Optional) Scale to Full 2.8M Kaggle Dataset
To index the entire 214,000+ AppleSupport corpus from Kaggle:
```bash
# Download full Kaggle dataset
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/ --unzip

# Reconstruct threads and rebuild BM25 index
python run_pipeline.py --rebuild-index
```

---

## Headline Results

Evaluated on 180 stratified golden test cases (cost penalty: $5 \times FN + 1 \times FP$):

| Metric Area | Metric | Trivial Baseline | Simple Baseline | Full System | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Classification** | Accuracy | 22.2% | 64.4% | **82.8%** | **+18.4%** |
| | Macro F1 | 0.045 | 0.669 | **0.838** | **+0.169** |
| **Escalation Decision** | Precision | 0.278 | 0.667 | **0.913** | **+0.246** |
| | Recall | **1.000** | 0.160 | **0.840** | **+0.680** |
| | $F_2$ Score | 0.658 | 0.189 | **0.854** | **+0.665** |
| | Cost Penalty | 130 | 214 | **44** | **-79.4% (Lower is better)** |
| **Reply Quality (Judge)** | Faithfulness | 3.00 / 5 | 4.53 / 5 | **5.00 / 5** | **+0.47** |
| | Tone Match | 4.00 / 5 | 4.47 / 5 | **4.97 / 5** | **+0.50** |
| | Resolves Issue | 2.00 / 5 | 3.70 / 5 | **4.50 / 5** | **+0.80** |
| | Brand Voice | 4.00 / 5 | 4.37 / 5 | **5.00 / 5** | **+0.63** |

*Human annotator agreement with LLM Judge across 30 items: Spearman $\rho = 0.652$, Quadratic Weighted $\kappa = 0.610$.*

---

## Dataset & Full Kaggle Pipeline

### Default Offline Mode
The repo includes `data/sample_apple.csv` and `data/golden_set.jsonl`, enabling immediate offline execution without downloading large files.

### Scaling to the Full 550MB Kaggle Dataset
To index all 214,000+ AppleSupport threads from the complete Kaggle dataset:

1. Download the Kaggle dataset using the Kaggle CLI:
   ```bash
   kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/ --unzip
   ```
2. Reconstruct threads and rebuild the BM25 index:
   ```bash
   python run_pipeline.py --rebuild-index
   ```

---

## CLI Usage

```bash
# 1. Process custom query
python run_pipeline.py --query "My iPhone screen is bulging out and getting very hot"

# 2. Interactive Ground Truth Labeling Tool
python data/labeler.py --input data/unlabeled_sample.jsonl

# 3. Judge vs. Human correlation analysis
python eval/human_agreement.py
```

---

## LLM Provider Configuration (Optional)

By default, the pipeline runs using a self-contained offline simulator for deterministic, zero-cost evaluation. To run with live LLM APIs, simply export your preferred provider's environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Or Google Gemini
export GEMINI_API_KEY="..."
```
The client automatically detects the active key and configures the corresponding model (`gpt-4o-mini`, `claude-3-5-haiku`, or `gemini-1.5-flash`).

---

## Repository Structure

```
hiver-support-agent/
├── README.md                  # Project overview, quickstart, architecture, headline results
├── REPORT.md                  # Comprehensive 6-page evaluation report (failure modes, critiques)
├── DECISIONS.md               # 13 non-obvious engineering decisions & tradeoffs
├── Makefile                   # make run, eval, test, label
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # Unified single-command pipeline runner
├── pytest.ini                 # Pytest configuration
│
├── config/
│   ├── taxonomy.yaml          # 8-intent definitions, routing, and few-shot examples
│   └── escalation_rules.yaml  # Safety hazard keywords, sentiment risk, thresholds
│
├── data/
│   ├── loader.py              # Thread reconstructor (Customer -> Brand -> Customer)
│   ├── sample_apple.csv       # Packaged stratified sample of AppleSupport threads
│   ├── golden_set.jsonl       # 180 stratified golden test cases
│   ├── human_eval_subset.jsonl# 30 cases with human ground truth rubric ratings
│   ├── retrieval_index.json   # Persistent BM25 search index
│   ├── NOTES.md               # Sampling methodology and annotation guidelines
│   └── labeler.py             # Interactive CLI labeling interface
│
├── src/
│   ├── agent.py               # AppleSupportAgent orchestrator
│   ├── intent_classifier.py   # Few-shot LLM classifier + regex baseline
│   ├── retrieval.py           # BM25 Okapi retrieval engine
│   ├── reply_generator.py     # Grounded AppleSupport reply drafter
│   ├── escalation.py          # Multi-factor auto-handle vs. escalate decision engine
│   ├── baselines.py           # Trivial and Simple baseline models
│   └── llm_client.py          # Provider-agnostic LLM interface (OpenAI, Anthropic, Gemini, Mock)
│
├── eval/
│   ├── evaluate.py            # Evaluation benchmark harness comparing system vs. baselines
│   ├── judge.py               # 4-criterion LLM-as-judge rubric evaluator
│   └── human_agreement.py    # Spearman rho & Cohen's kappa correlation script
│
└── tests/
    ├── test_loader.py         # Thread reconstruction tests
    ├── test_retrieval.py      # BM25 index & persistence tests
    ├── test_classifier.py     # Intent classification tests
    ├── test_escalation.py     # Escalation rule & threshold tests
    └── test_agent.py          # End-to-end pipeline integration tests
```

---

## License
MIT License. Created for the Take-Home AI Customer Support Agent Assessment.
