# NBA Text-to-SQL Final Presentation (8-10 min)

## Slide 1 — Problem and Goal
- Task: translate NBA natural language questions to executable SQL.
- Why it matters: real analytics workflows depend on robust schema grounding.
- Project type: Applied track, Training focus.

## Slide 2 — System Overview
- Pipeline: question -> schema context (full/oracle/RAG) -> seq2seq model -> SQL -> SQLite execution.
- Components: Spider pretraining, NBA adaptation, PEFT training, held-out evaluation.

## Slide 3 — Data and Split Protocol
- Spider as source domain.
- NBA custom set: 200 question/SQL pairs.
- Deterministic 150/50 train-test split to avoid leakage.

## Slide 4 — Methods Compared
- Full fine-tune, LoRA, QLoRA.
- Model families: T5-base and CodeT5+ (220M).
- Adaptation curve setup: n=0/20/70/all.

## Slide 5 — Main Results (Table)
- Show `eval/results_summary.md` key rows.
- Highlight best run: `lora_codet5p-220m_r16_nba_nall_nba_oracle_test`.
- Emphasize exec accuracy as primary practical metric.
- Add a compact Spider control table from `eval/spider_summary.md` (CodeT5 LoRA r16 > full T5 > QLoRA T5 > LoRA T5 ranks).

## Slide 6 — Cross-domain Adaptation Curve
- Plot from `eval/fewshot_curve.csv`.
- Message: adaptation data drives the largest gains.
- Mention added n=10 support for consistent curve reporting.

## Slide 7 — RAG Ablation
- Compare oracle vs RAG k=1/3/5 from `eval/rag_ablation.csv`.
- Message: retrieval quality is current bottleneck in this stack.

## Slide 8 — Error Analysis
- Taxonomy: schema_linking / structural / value / aggregation.
- Show top 2 dominant failure classes for best and baseline models.
- Explain one concrete failure example.

## Slide 9 — Demo and Reproducibility
- Live demo flow via `src/demo.py`.
- Reproducibility artifacts:
  - `run_config.json` per run
  - fixed split
  - one-command matrix generator
  - auto-generated summary CSV/MD tables

## Slide 10 — Takeaways and Next Steps
- What worked: LoRA + domain adaptation.
- What failed: current RAG retrieval.
- Next: stronger retriever, multi-seed confidence intervals, expanded NBA benchmark.
