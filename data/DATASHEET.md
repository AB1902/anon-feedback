# Datasheet: Data used in the Anonymous Feedback System

Use this section directly in the "Dataset" part of the report. Fill in the
bracketed numbers after you run the generator (the split script prints them).

## 1. Summary

| Component | Source | Size | Purpose |
|---|---|---|---|
| Company structure | Synthetic (`scripts/build_company.py`) | 255 employees, 8 departments, 21 teams | Gives the dashboards real org structure; supplies fake names/IDs to plant as PII |
| Employee feedback | Synthetic, LLM-generated (`scripts/generate_feedback.py`) | 1,000 items | Training/evaluation data for triage, privacy layer and dashboards |
| Adversarial set | Hand-written templates (`scripts/build_adversarial.py`) | 110 items | Red-team evaluation of guardrails |
| HR policy handbook | Synthetic, drafted for this project (`data/handbook/handbook.md`) | 15 sections, ~90 numbered clauses | RAG corpus for policy-grounded explanations |
| Real-world reference (optional) | AmbitionBox / Glassdoor reviews | sample | Qualitative check that the classifier works on real employee language |
| Privacy evaluation (optional) | ai4privacy/pii-masking-400k (English) | sample | External benchmark for PII scrubber recall |
| Injection evaluation (optional) | deepset/prompt-injections | ~660 | External benchmark for injection detection |

## 2. Why synthetic data
Real internal employee feedback is confidential by nature and cannot ethically be
collected for a student project. Synthetic data also gives **perfect ground truth**:
every item is generated from a specification (category, sentiment, severity,
planted PII), so accuracy and PII recall can be measured exactly.

## 3. Generation method
1. A deterministic grid samples, for each item: department (weighted by headcount),
   level band (75% IC, 20% manager, 5% senior leadership), ISO week (26-week window),
   category, sentiment, severity, and one of 7 writing styles
   (formal, angry, Hinglish, short, rambling, polite, sarcastic).
2. **Severity rules:** harassment, safety, fraud and HR-conduct items are always
   negative with severity 3–5. Other categories are 65% negative / 20% neutral /
   15% positive.
3. **Planted PII:** about 70% of items get 1–3 identifiers taken from the fake
   employee roster (full or first names, employee IDs, emails, phone numbers,
   team + location such as "Pune Payments team", project code names). The generator
   verifies each planted value actually appears in the output text.
4. **Storylines** (so the dashboard has real trends to find):
   - Sales: compensation complaints (incentive delays) rise sharply in the last 8 weeks.
   - Engineering: workload complaints (release crunch) rise in the last 6 weeks.
   - Customer Support: workload complaints decline over time (improving team).
5. Items are written by an LLM (Gemini Flash / Groq) in batches of 8 at low
   temperature, using the prompt in `generate_feedback.py`.
6. **Split:** stratified by category, 80% train / 20% held-out test.

## 4. Labels (ground truth)

| Field | Values |
|---|---|
| category | compensation, manager_behaviour, workload, culture, growth, other, harassment, safety, fraud, hr_conduct |
| sentiment | positive, neutral, negative |
| severity | 1–5 |
| escalate / route | derived from category: harassment → POSH committee; safety, fraud → Ethics committee; hr_conduct → Senior review panel; others → HR |
| pii | list of {type, value} actually present in the text |

## 5. Adversarial set composition

| Type | Count | Expected |
|---|---|---|
| prompt_injection | 38 | block |
| embedded_injection (genuine feedback + attack) | 9 | block |
| deanonymisation attempts | 13 | block |
| abuse / threats | 12 | block |
| off-topic / spam | 15 | block |
| benign look-alikes (contain "ignore", "override", etc.) | 23 | allow |

The benign look-alikes measure false positives, so a guardrail that blocks
everything scores badly.

## 6. Final counts (fill in after generation)
- Items generated: [ ]   Train: [ ]   Test: [ ]
- PII values planted and present: [ ]   Dropped by LLM: [ ]

## 7. Limitations
- Synthetic text is cleaner and more uniform than real feedback; results on the
  test set are likely optimistic. The optional real-review check addresses this.
- Labels come from the generation spec; an LLM may occasionally write text that
  drifts from its spec. Spot-check ~50 items manually and report the agreement rate.
- Free-tier LLM APIs may use prompts to improve their models. Only synthetic data
  is ever sent; a production deployment would use a paid or self-hosted model.
- The handbook is fictional and simplified; it references Indian laws (POSH Act 2013,
  Maternity Benefit Act 1961, DPDP Act 2023) for realism only and is not legal advice.
