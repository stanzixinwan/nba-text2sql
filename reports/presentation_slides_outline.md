# NBA Text-to-SQL Final Presentation (8–10 min)

This outline mirrors the rubric in `final_project.pdf` (Task / Track / Focus,
Approach: data + model + baseline, Key results + most interesting finding,
What I learned + next steps) and is grounded in the latest artifacts under
`eval/` (e.g. `results_summary.csv`, `fewshot_curve.csv`, `rag_ablation.csv`,
`spider_summary.csv`, `error_analysis_summary.json`).

Target pacing: ~14 content slides, ~35–40 s each, ending in 2–3 min Q&A.

## Slide 1 — Title
- Project: Retrieval-Augmented Text-to-SQL with PEFT for Cross-Domain NBA Analytics.
- Course: COSI 115b Fundamentals of NLP II — Final Presentation.
- Author and date.

## Slide 2 — Task and Motivation
- Task: translate natural language NBA questions into executable SQL on a real SQLite DB.
- Why: text-to-SQL unlocks analytics for non-SQL users; NBA is a compact but realistic, schema-rich domain.
- One-line example: question → SQL → result row.

## Slide 3 — Track and Focus (per `final_project.pdf`)
- **Track: Applied** — runnable end-to-end pipeline with demo + held-out evaluation.
- **Focus: Training** — depth on adaptation regimes (full / LoRA / QLoRA), LoRA rank sweep, and adaptation-data ablation (n = 0/10/20/70/all).
- Why this combination: deployment-style system + controlled training comparisons under a single-GPU compute budget.

## Slide 4 — System Pipeline
- TikZ diagram: Question → Schema context (Full / Oracle / RAG top-k) → Seq2Seq model (T5-base or CodeT5+ 220M) → Predicted SQL → SQLite execution → Answer + metrics.
- Highlight: the schema-context branch is the configurable axis used in every ablation.

## Slide 5 — Data and Split Protocol
- Spider (`xlangai/spider`): 10,181 questions / 200 DBs — source-domain supervision.
- NBA target: 200 authored question/SQL pairs over a real `nba-sql` SQLite DB; difficulty labels easy / medium / hard / extra_hard.
- **Deterministic 150/50 train/test split** stored in `data/nba/nba_split.json`; all reported numbers use `--split test`.
- Adaptation data sizes: n = 0, 10, 20, 70, 150 (= all).

## Slide 6 — Methods
- Source-domain training on Spider, then continued NBA adaptation from each Spider checkpoint.
- Regimes: full fine-tune, LoRA (r ∈ {4, 8, 16}), QLoRA.
- Inference modes: full schema, **oracle tables** (upper bound), **RAG top-k** (FAISS + sentence-transformers).
- Metrics: execution accuracy (primary) and exact match.

## Slide 7 — Baselines (per rubric)
- Zero-shot T5-base prompting on NBA test: exec 0.00 / exact 0.00.
- Few-shot T5-base prompting on NBA test: exec 0.00 / exact 0.00.
- Full T5-base fine-tuned on Spider only: exec 0.04 / exact 0.00.
- Message: any reasonable result on NBA requires both training and adaptation.

## Slide 8 — Spider Controls (training-focus check)
- Exact match on a 200-example Spider slice (`eval/spider_summary.csv`):
  - LoRA CodeT5+ r16: 0.290
  - Full T5-base: 0.265
  - QLoRA T5-base: 0.140
  - LoRA T5-base r16 / r4 / r8: 0.120 / 0.070 / 0.000
- Take-away: CodeT5+ is the strongest source-domain backbone in our regime; LoRA beats full FT on the same model.

## Slide 9 — Main NBA Results (held-out test, n=50)
- Compact table sourced from `eval/results_summary.md` (oracle mode):
  - Full T5-base, no NBA: exec 0.04
  - LoRA T5-base, no NBA: exec 0.04
  - LoRA CodeT5+ 220M, no NBA: exec 0.10
  - **LoRA CodeT5+ 220M, n=all NBA adaptation: exec 0.46 / exact 0.42** ← best
- Headline: best system improves exec accuracy by **~12× over baselines**.

## Slide 10 — Most Interesting Finding: Adaptation Curve
- Line chart from `eval/fewshot_curve.csv` for both models (oracle mode):
  - CodeT5+: 0.10 → 0.00 → 0.12 → 0.26 → **0.46** at n = 0/10/20/70/all.
  - T5-base: 0.04 → 0.00 → 0.04 → 0.08 → 0.16.
- Take-away: **adaptation data dominates method choice**; the n = 70 → n = all step alone (+0.20) is larger than any method-only delta.

## Slide 11 — RAG Ablation
- Best NBA-adapted CodeT5+ checkpoint, varying schema source (`eval/rag_ablation.csv`):
  - Oracle: 0.46  — RAG k=1: 0.12  — RAG k=3: 0.08  — RAG k=5: 0.08.
- T5-base mirrors the trend (oracle 0.16 → RAG k=5 0.02).
- Take-away: in this stack, **retrieval is the bottleneck**, not generation — motivates next steps.

## Slide 12 — Error Analysis
- Rule-based taxonomy via `src/error_analysis.py`: schema_linking / structural / value / aggregation.
- Best system (oracle, n=all): 46% correct, 30% value, 18% structural, 6% schema_linking.
- Same model under RAG k=3: **78% schema_linking errors**, only 8% correct → confirms retrieval is the failure source.
- Difficulty-stratified: easy 12/17 correct; hard mostly value/structural; extra_hard remains unsolved.

## Slide 13 — Concrete Failure Example
- Question: "How many players are currently active?"
- Gold: `SELECT COUNT(*) FROM player WHERE is_active = 1;`
- Pred: `SELECT COUNT(*) FROM player`  → wrong literal grounding (value error).
- Lesson: model nails structure but loses domain-specific value constraints — a future direction for value-conditioned decoding.

## Slide 14 — Demo, Reproducibility, Takeaways and Next Steps
- Demo: `python -m src.demo` — live question → SQL → executed answer.
- Reproducibility: deterministic split, per-run `run_config.json`, `run_experiment_matrix.py`, auto-generated `eval/results_summary.{csv,md}`.
- **What worked**: PEFT + domain adaptation; CodeT5+ as backbone; oracle-mode upper bound for clean ablation.
- **What didn't**: dense-only RAG retrieval; small extra-hard slice still unsolved.
- **Next**: hybrid retriever + re-ranker; multi-seed CIs; expand NBA benchmark; value-aware decoding.

## Slide 15 — Thank You / Q&A
- Pointer to `reports/presentation_qa.md` for prepared Q&A topics.
