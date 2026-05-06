# Eval Folder Guide

This folder contains model evaluation outputs and derived analysis artifacts.

## 1) Core NBA test results (used in final tables)

- `baseline_nba_zeroshot_t5-base_test.json`
- `baseline_nba_fewshot_t5-base_test.json`
- `full_t5-base_nba_oracle_test.json`
- `lora_t5-base_r16_nba_oracle_test.json`
- `lora_t5-base_r16_nba_n20_nba_oracle_test.json`
- `lora_t5-base_r16_nba_n70_nba_oracle_test.json`
- `lora_t5-base_r16_nba_nall_nba_oracle_test.json`
- `lora_t5-base_r16_nba_nall_nba_rag_k5_test.json`
- `lora_codet5p-220m_r16_nba_oracle_test.json`
- `lora_codet5p-220m_r16_nba_n20_nba_oracle_test.json`
- `lora_codet5p-220m_r16_nba_n70_nba_oracle_test.json`
- `lora_codet5p-220m_r16_nba_nall_nba_oracle_test.json`
- `lora_codet5p-220m_r16_nba_nall_nba_rag_k1_test.json`
- `lora_codet5p-220m_r16_nba_nall_nba_rag_k3_test.json`
- `lora_codet5p-220m_r16_nba_nall_nba_rag_k5_test.json`

## 2) Aggregated outputs (auto-generated)

- `results_summary.csv`
- `results_summary.md`
- `fewshot_curve.csv`
- `rag_ablation.csv`
- `missing_results.csv`
- `error_analysis_summary.json`

## 3) Annotated / analysis-only artifacts

- `lora_codet5p-220m_r16_nba_nall_nba_oracle_test_annotated.json`
- `lora_codet5p-220m_r16_nba_nall_nba_rag_k3_test_annotated.json`

## 4) Naming convention

- `*_test.json`: held-out NBA test split (50 examples)
- `*_oracle*`: oracle-table schema mode
- `*_rag_k{n}*`: RAG retrieval mode with top-k tables
- `*_nba_n{n}*`: NBA adaptation train size (`n20`, `n70`, `nall`, etc.)
