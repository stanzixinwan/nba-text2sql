# Current Gaps and Limitations (as of now)

## 1) Missing planned experiment outputs

Based on `eval/missing_results.csv`, these expected files are still missing:

- `lora_t5-base_r4_nba_oracle_test.json`
- `lora_t5-base_r8_nba_oracle_test.json`
- `lora_t5-base_r32_nba_oracle_test.json`
- `qlora_t5-base_nba_oracle_test.json`
- `lora_t5-base_r16_nba_n10_nba_oracle_test.json`
- `lora_t5-base_r16_nba_nall_nba_rag_k1_test.json`
- `lora_t5-base_r16_nba_nall_nba_rag_k3_test.json`

Implication:
- Rank sweep is incomplete for T5 (`r4/r8/r32` missing).
- Method comparison lacks QLoRA test result.
- Few-shot adaptation curve lacks `n=10` point.
- T5 RAG ablation only has `k=5`, missing `k=1/3`.

## 2) Runtime environment bottlenecks

- Current non-interactive runs in this session defaulted to CPU for baseline evaluation, which is much slower.
- `models/` currently has no checked local checkpoints in this workspace snapshot, so some adaptation/rank experiments cannot be resumed from local files and must be retrained.

Implication:
- Full matrix completion can take multiple hours to >1 day on single GPU.
- If GPU is not visible in the execution environment, experiments are significantly delayed.

## 3) Statistical rigor still limited

- Most reported numbers are single-run values (no multi-seed confidence intervals).
- No significance test is reported between key variants.

Implication:
- Conclusions are directionally useful but not yet statistically strong for publication-quality claims.

## 4) Retrieval pipeline weaknesses

- Existing RAG results for CodeT5+ show performance drop vs oracle schema.
- Dominant RAG error category is schema linking (from `error_analysis` output).

Implication:
- Retrieval quality is currently the main bottleneck, not purely decoder capacity.

## 5) Immediate next commands to close gaps

Recommended command sequence (GPU terminal):

```bash
# (A) Rank sweep missing points
python -m src.train --method lora --model t5-base --rank 4 --epochs 3 --lr 1e-4 --seed 42
python -m src.train --method lora --model t5-base --rank 8 --epochs 3 --lr 1e-4 --seed 42
python -m src.train --method lora --model t5-base --rank 32 --epochs 3 --lr 1e-4 --seed 42

# Evaluate each rank checkpoint on NBA test/oracle
python -m src.evaluate --checkpoint models/lora_t5-base_r4_lr0.0001_s42/final --eval nba --oracle-tables --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r8_lr0.0001_s42/final --eval nba --oracle-tables --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r32_lr0.0001_s42/final --eval nba --oracle-tables --split test

# (B) QLoRA comparison
python -m src.train --method qlora --model t5-base --rank 16 --epochs 3 --lr 1e-4 --seed 42
python -m src.evaluate --checkpoint models/qlora_t5-base_lr0.0001_s42/final --eval nba --oracle-tables --split test

# (C) n=10 adaptation point
python -m src.train_nba --base-checkpoint models/lora_t5-base_r16_lr0.0001_s42/final --base-model t5-base --n-train 10 --seed 42
python -m src.evaluate --checkpoint models/lora_t5-base_r16_lr0.0001_s42_nba_n10_s42/final --eval nba --oracle-tables --split test

# (D) T5 RAG missing k
python -m src.evaluate --checkpoint models/lora_t5-base_r16_nba_nall_s42/final --eval nba --use-rag --top-k 1 --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r16_nba_nall_s42/final --eval nba --use-rag --top-k 3 --split test

# (E) Rebuild summary tables and plots inputs
python -m src.audit_missing_results --eval-dir eval
python -m src.build_results_report --eval-dir eval
```
