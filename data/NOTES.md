# Dataset Sampling & Labeling Notes (NOTES.md)

## 1. Sampling Methodology & Stratification

To ensure unbiased and representative evaluation across the operational spectrum of **@AppleSupport**, we curated the evaluation datasets using a **stratified random sampling** strategy:

### Stratification Dimensions:
1. **Intent Distribution (8 Classes)**:
   - `software_issue` (25%): OS glitches, freezes, app crashes, iOS update regressions.
   - `hardware_battery` (18%): Battery degradation, screen lifting/cracks, microphone/speaker faults.
   - `account_security` (12%): Locked Apple ID, 2FA prompt spam, stolen devices, unauthorized logins.
   - `billing_subscription` (12%): In-app purchase disputes, recurring charges, card declination.
   - `how_to_inquiry` (12%): Pairing AirPods, backup restoration, settings toggles.
   - `order_repair_status` (10%): Genius Bar depot repair IDs, Apple Store shipment tracking.
   - `complaint_frustration` (8%): Hostile complaints, threats of legal action/churn, repeated failures.
   - `spam_or_irrelevant` (3%): Memes, trolling, empty greetings, unrelated mentions.

2. **Temporal Coverage**:
   - Spans early iOS versions (iOS 10 / 11 release windows) and varied time-of-day/day-of-week customer activity to avoid seasonal bias.

3. **Thread Depth & Complexity**:
   - Single-turn direct queries vs. multi-turn ongoing troubleshooting contexts.

---

## 2. Golden Dataset Schema (`golden_set.jsonl`)

Each golden record in `golden_set.jsonl` contains:
```json
{
  "id": "gold_001",
  "customer_tweet": "Ever since updating to iOS 11.1 my battery drops 20% in 15 minutes and apps keep stuttering.",
  "gold_intent": "software_issue",
  "gold_action": "auto",
  "gold_reasoning": "Standard post-update OS glitch. Auto-handle with clarifying questions (iOS build, battery usage by app) and standard reboot/diagnostic guidance.",
  "escalation_triggers": [],
  "urgency": "medium",
  "created_at": "2017-10-15 14:22:00"
}
```

---

## 3. Ground Truth Annotation Guidelines

### A. Intent Labeling Criteria
- Classify based on the **primary root cause** or primary assistance requested by the customer.
- If a customer mentions an update followed by battery drain, classify as `software_issue` if tied to a recent OS update, or `hardware_battery` if asking about battery replacement / health percentage under 80%.
- If a customer expresses anger while mentioning a repair status, classify as `complaint_frustration` if their primary statement is about service failure/unacceptable delay, or `order_repair_status` if they are primarily seeking the tracking update.

### B. Escalation Criteria (Auto vs. Escalate)
- **Escalate (`escalate`)** MUST be chosen if ANY of the following are true:
  1. **Security & Identity**: Customer reports compromised Apple ID, unauthorized charges, stolen phone, 2FA lockouts, or activation lock on used devices.
  2. **Physical Safety Hazards**: Battery bulging/swelling, smoke, sparks, melting, electric shock.
  3. **High Churn / Legal Risk**: Customer threatens lawsuit, BBB complaint, or explicitly requests managerial escalation / compensation.
  4. **Exhausted First-Line**: Customer explicitly states they have already completed reboots, resets, and Genius Bar visits with no resolution.
- **Auto-Handle (`auto`)** is chosen when:
  - The query can be addressed via public support URLs, standard diagnostic questions (device model, iOS version), or standard navigational guidance (e.g., Settings > Battery).

---

## 4. Cost Asymmetry & Evaluation Metrics
In customer support automation:
- **False Negative (Missed Escalation)**: Severe business harm. An angry customer, hacked account, or exploding battery is left with a generic bot response.
- **False Positive (Unnecessary Escalation)**: Mild cost. A human agent reviews an inquiry that could have been handled automatically.
- **Cost Formula**:
  $$\text{Total Escalation Penalty} = (5 \times \text{False Negatives}) + (1 \times \text{False Positives})$$
- We report **Precision**, **Recall**, and **$F_2$ score** (prioritizing recall over precision) alongside the cost matrix.
