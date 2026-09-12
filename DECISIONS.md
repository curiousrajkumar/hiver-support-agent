# Engineering Decisions & Tradeoffs (DECISIONS.md)

This document details 13 non-obvious engineering decisions, architectural choices, and explicit tradeoffs made while building the **@AppleSupport AI Support Agent**.

---

### 1. Brand Selection: Why @AppleSupport over AmazonHelp or Delta
* **Decision:** Selected `@AppleSupport` as the single brand focus.
* **Rationale:** AppleSupport is the highest-volume technology support handle in the Kaggle dataset (>214,000 tweets). Unlike retail order inquiries (`@AmazonHelp`) or flight rescheduling (`@Delta`), Apple queries span multiple complex technical layers: hardware degradation, iOS update regressions, iCloud sync, Apple ID/security lockouts, and App Store billing disputes.
* **Tradeoff:** Apple enforces strict privacy rules (never handling serial numbers or Apple IDs in public tweets), requiring our pipeline to model diagnostic triage and DM escalation handoffs rather than fully resolving sensitive issues on public Twitter.

---

### 2. Intent Taxonomy: 8 Discrete Intents vs. Hierarchical or Coarse Clustering
* **Decision:** Fixed the taxonomy to 8 mutually exclusive intents: `software_issue`, `hardware_battery`, `account_security`, `billing_subscription`, `how_to_inquiry`, `order_repair_status`, `complaint_frustration`, and `spam_or_irrelevant`.
* **Rationale:** Coarse clustering (e.g., 3-4 classes: Tech, Billing, Other) obscures vital downstream routing distinctions (e.g., treating a bulging battery the same as an iOS calculator lag). Conversely, 20+ micro-intents introduce heavy label overlap and annotation ambiguity on 280-character tweets.
* **Tradeoff:** Certain multi-part tweets (e.g., "iOS 11 update drained my battery and your technician was rude") exhibit intent overlap. We established clear prioritization guidelines: severe frustration and safety take precedence over general software bugs.

---

### 3. Classification Engine: Few-Shot LLM with JSON Schema vs. Fine-Tuned Model
* **Decision:** Built the primary intent classifier as a few-shot LLM with strict JSON schema output and explanatory reasoning, backed by a deterministic regex/keyword fallback.
* **Rationale:** A fine-tuned lightweight model (e.g., DistilBERT/RoBERTa) requires extensive offline training, GPU resources, and checkpoint management, violating the <15 min laptop reproduction constraint. Few-shot LLM prompting delivers immediate zero-training deployment, handles colloquial Twitter abbreviations and slang gracefully, and yields transparent reasoning for human auditability.
* **Tradeoff:** Inference latency for live LLM API calls is ~300-800ms compared to ~15ms for local BERT embeddings. For high-volume production, a distilled student model could be trained on our golden set.

---

### 4. Retrieval Index: BM25 Okapi with Local Inverted Index vs. Dense Embeddings (FAISS)
* **Decision:** Implemented a self-contained, reproducible BM25 Okapi inverted index with JSON disk serialization.
* **Rationale:** Dense embedding models (e.g., Sentence-Transformers / text-embedding-3-small) require heavy PyTorch/ONNX dependencies and frequently hallucinate similarity on version numbers and error codes (e.g., treating `iOS 11.0.1` and `iOS 10.3.3` as identical semantic vectors). BM25 provides exact keyword and model token matching, sub-millisecond querying, zero external API costs, and runs in <1 second on any laptop.
* **Tradeoff:** BM25 cannot capture deep semantic paraphrasing where no vocabulary overlaps (e.g., "my handset died" vs. "iPhone shut down"). We compensated by cleaning and expanding the historical corpus index.

---

### 5. Thread Reconstruction Heuristic ($T_1 \to T_2 \to T_3$)
* **Decision:** Linked tweets using `in_response_to_tweet_id` to construct clean `(customer_initiating_tweet, brand_reply, customer_followup)` tuples.
* **Rationale:** The raw Kaggle dataset is a flat CSV of 2.8M rows. Reconstructing conversational context ensures the retrieval index indexes only *customer problem descriptions* mapped to *official AppleSupport replies*, preventing brand-to-brand administrative replies from contaminating the corpus.
* **Tradeoff:** Dropped multi-party conversational branches where third-party users chimed into an AppleSupport thread, focusing strictly on dyadic customer-brand pairs.

---

### 6. Multi-Factor Escalation Engine vs. Single-Pass Prompt
* **Decision:** Implemented escalation as a separate multi-factor deterministic and probabilistic rule engine (`src/escalation.py`) that evaluates safety keywords, sentiment risk, retrieval distance, and intent ambiguity.
* **Rationale:** Asking a generative LLM in a single pass to both draft a polite reply and decide whether to escalate leads to prompt leakage and sycophancy (the LLM often attempts to auto-solve critical safety or legal hazards because it is trained to be helpful). A decoupled policy engine guarantees 100% deterministic interception of physical hazards (bulging batteries, smoke) and legal/regulatory threats.
* **Tradeoff:** Requires explicit maintenance of keyword and pattern dictionaries in `config/escalation_rules.yaml`.

---

### 7. Cost Asymmetry: 5x Penalty for Missed Escalation ($C_{fn} = 5, C_{fp} = 1$)
* **Decision:** Evaluated escalation accuracy using an asymmetric cost matrix where False Negatives (missed escalation) are penalized 5x higher than False Positives (unnecessary escalation).
* **Rationale:** In customer support operations, auto-replying to an exploding battery, a hacked Apple ID, or an angry lawsuit threat with a canned troubleshooting guide causes severe safety risks, brand damage, or customer churn. Conversely, an unnecessary escalation only costs 2-3 minutes of human agent triage time.
* **Tradeoff:** Pushes the model toward a slightly conservative bias, but dramatically reduces catastrophic operational failures.

---

### 8. Primary Escalation Metric: $F_2$ Score over $F_1$ Score
* **Decision:** Adopted the $F_2$ measure ($\beta = 2$) as the headline escalation performance metric.
* **Rationale:** Standard $F_1$ weights precision and recall equally. Because missed escalations are operationally unacceptable, $F_2$ gives twice the weight to recall over precision, rewarding models that capture true escalations even if precision dips slightly.
* **Tradeoff:** A model that aggressively escalates might report a high $F_2$ while increasing human queue volume; we balance this by tracking Cost-per-Query alongside $F_2$.

---

### 9. Grounded Few-Shot In-Context Reply Generation
* **Decision:** Injected the top-3 retrieved historical pairs directly into the LLM system prompt as grounding context before drafting.
* **Rationale:** Prevents hallucinated troubleshooting steps and keeps the agent strictly within the authentic @AppleSupport communication style (e.g., asking for iOS version, pointing to `Settings > General > About`, directing billing disputes to `https://reportaproblem.apple.com`).
* **Tradeoff:** In-context grounding consumes ~400-600 prompt tokens per reply generation.

---

### 10. Decomposed LLM-as-Judge Rubric (4 Discrete 1-5 Dimensions)
* **Decision:** Scored replies across 4 explicit criteria: Grounding Faithfulness, Tone Match, Resolves the Issue, and Brand Voice Consistency.
* **Rationale:** Asking an LLM judge for a single holistic 1-5 score causes "halo bias" (a polite response gets a 5 even if the technical troubleshooting advice is completely wrong). Decomposing the evaluation forces the judge to separately evaluate factual grounding and tonal empathy.
* **Tradeoff:** Requires structured JSON parsing and higher judge evaluation latency.

---

### 11. Human Agreement Validation: Spearman Rank & Cohen's Kappa
* **Decision:** Evaluated the LLM judge against 30 human-annotated test cases using Spearman rank correlation ($\rho$) and quadratic weighted Cohen's Kappa ($\kappa$).
* **Rationale:** An LLM judge cannot be trusted in production without measuring inter-rater reliability against human standards. Quadratic weighted kappa accounts for ordinal distance (penalizing a 1 vs. 5 disagreement far more than a 4 vs. 5 disagreement).
* **Tradeoff:** A 30-sample human subset is small for tight statistical significance bounds, but provides an effective sanity check on directional calibration.

---

### 12. Provider-Agnostic LLM Interface with Offline Fallback
* **Decision:** Designed `src/llm_client.py` to auto-detect `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`, and provide a high-fidelity deterministic offline mock simulator when no keys are set.
* **Rationale:** Take-home reviewers and CI pipelines must be able to run `python run_pipeline.py` or `make eval` immediately without hitting paywalls, rate limits, or provisioning API keys.
* **Tradeoff:** The offline mock uses semantic heuristic templates rather than generating novel generative prose, but mirrors the exact input/output contract and scoring distributions.

---

### 13. Bundled Stratified Dataset vs. Mandatory 550MB Kaggle Download
* **Decision:** Packaged `data/sample_apple.csv` (258 representative tweet threads) and `data/golden_set.jsonl` (180 stratified golden items) directly in the repository.
* **Rationale:** The full Kaggle `twcs.csv` is ~550MB (2.8M rows). Downloading and parsing it on a laptop exceeds the 15-minute evaluation window. Bundling the sample ensures sub-second initialization while `data/loader.py` provides full Kaggle CLI support for production scaling.
* **Tradeoff:** The offline BM25 index contains ~130 historical threads rather than 214,000, but fully demonstrates the indexing, retrieval, and grounding architecture.
