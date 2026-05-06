# Missing Results: Actions

Source of truth: `missing_results.csv`

## Missing files and whether training is required

1) `lora_t5-base_r4_nba_oracle_test.json`  
- Need training? **Yes** (if checkpoint `models/lora_t5-base_r4_*` not already available)

2) `lora_t5-base_r8_nba_oracle_test.json`  
- Need training? **Yes** (if checkpoint `models/lora_t5-base_r8_*` not already available)

3) `lora_t5-base_r32_nba_oracle_test.json`  
- Need training? **Yes** (if checkpoint `models/lora_t5-base_r32_*` not already available)

4) `qlora_t5-base_nba_oracle_test.json`  
- Need training? **Yes**

5) `lora_t5-base_r16_nba_n10_nba_oracle_test.json`  
- Need training? **Yes** (NBA adaptation `n=10` must be trained first)

6) `lora_t5-base_r16_nba_nall_nba_rag_k1_test.json`  
- Need training? **No (usually)**, evaluate existing `nall` checkpoint with `--top-k 1`

7) `lora_t5-base_r16_nba_nall_nba_rag_k3_test.json`  
- Need training? **No (usually)**, evaluate existing `nall` checkpoint with `--top-k 3`

If your local/Colab environment does not have the referenced checkpoint, then evaluation-only items become train+eval.

---

## Colab command block (copy-paste)

```bash
# 0) Activate env and move to repo
# cd /content/drive/MyDrive/nba-text2sql

# 1) Rank sweep missing points (train)
python -m src.train --method lora --model t5-base --rank 4 --epochs 3 --lr 1e-4 --seed 42
python -m src.train --method lora --model t5-base --rank 8 --epochs 3 --lr 1e-4 --seed 42
python -m src.train --method lora --model t5-base --rank 32 --epochs 3 --lr 1e-4 --seed 42

# 2) Evaluate rank sweep checkpoints on NBA test/oracle
python -m src.evaluate --checkpoint models/lora_t5-base_r4_lr0.0001_s42/final --eval nba --oracle-tables --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r8_lr0.0001_s42/final --eval nba --oracle-tables --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r32_lr0.0001_s42/final --eval nba --oracle-tables --split test

# 3) QLoRA (train + eval)
python -m src.train --method qlora --model t5-base --rank 16 --epochs 3 --lr 1e-4 --seed 42
python -m src.evaluate --checkpoint models/qlora_t5-base_lr0.0001_s42/final --eval nba --oracle-tables --split test

# 4) NBA n=10 adaptation (train + eval)
python -m src.train_nba --base-checkpoint models/lora_t5-base_r16_lr0.0001_s42/final --base-model t5-base --n-train 10 --seed 42
python -m src.evaluate --checkpoint models/lora_t5-base_r16_lr0.0001_s42_nba_n10_s42/final --eval nba --oracle-tables --split test

# 5) T5 RAG missing k values (eval-only if nall checkpoint exists)
python -m src.evaluate --checkpoint models/lora_t5-base_r16_nba_nall_s42/final --eval nba --use-rag --top-k 1 --split test
python -m src.evaluate --checkpoint models/lora_t5-base_r16_nba_nall_s42/final --eval nba --use-rag --top-k 3 --split test

# 6) Refresh summaries
python -m src.audit_missing_results --eval-dir eval
python -m src.build_results_report --eval-dir eval
```
