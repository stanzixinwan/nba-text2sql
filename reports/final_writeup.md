# Retrieval-Augmented Text-to-SQL with PEFT for Cross-Domain NBA Analytics

**Course:** COSI 115b Fundamentals of NLP II  
**Track / Focus:** Applied + Training  
**Author:** Zixin (Stan) Wan

## 1. Introduction

This project builds an end-to-end text-to-SQL system that maps natural language questions to executable SQL queries. The main practical target is a domain-specific NBA analytics database, while the primary training source is Spider. This setup intentionally creates a cross-domain transfer problem: training data is broad and heterogeneous, while deployment questions are from a focused sports schema with distinct vocabulary and query patterns.

The project follows an Applied track because the deliverable is a runnable pipeline (input question -> SQL generation -> SQL execution -> answer), not only an offline model benchmark. The depth focus is Training: I compare multiple adaptation strategies (full fine-tuning, LoRA, QLoRA), parameter settings (LoRA rank), and domain-adaptation data regimes (0/20/70/all NBA training examples, with n=10 pipeline support added for final experiments).

Two practical questions drive the work:

1. Which parameter-efficient strategy gives the best tradeoff for domain transfer into NBA SQL?
2. Does schema retrieval (RAG over table schemas) improve downstream SQL correctness in this setup?

## 2. Related Work

Text-to-SQL was standardized by Spider (Yu et al., 2018), which emphasizes compositional generalization across unseen databases. Modern approaches typically combine pretrained seq2seq models with schema conditioning, then evaluate by exact match and execution accuracy.

Parameter-efficient fine-tuning (PEFT), especially LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023), enables lower-memory adaptation and faster iteration versus full-model updates. For practical deployment settings with limited compute, PEFT is often the only feasible path to run ablations.

RAG (Lewis et al., 2020) motivates selective context retrieval when full schema context is too long. For text-to-SQL, retrieval can reduce irrelevant schema tokens, but retrieval mistakes may propagate to generation errors.

This project combines these lines in one applied workflow: Spider pretraining, NBA domain adaptation, PEFT comparisons, and schema retrieval ablations on a held-out NBA split.

## 3. Data

### 3.1 Spider (source domain)

- Dataset: `xlangai/spider` from HuggingFace
- Scale: 10,181 questions, 200 databases, 138 domains
- Role: source-domain supervised training for text-to-SQL patterns

### 3.2 NBA (target domain)

- Database: SQLite exported from `nba-sql` pipeline
- QA dataset: `data/nba/nba_questions.json` (200 authored question/SQL pairs)
- Difficulty labels: `easy`, `medium`, `hard`, `extra_hard`
- Target tables include player/team/game and supporting relation tables

### 3.3 Train/test protocol

NBA evaluation is standardized through a deterministic 150/50 split (`data/nba/nba_split.json`):

- Train partition: used for domain adaptation experiments (`n=10/20/70/all`)
- Test partition: fixed held-out set for final reporting

This avoids leakage and makes every comparison directly comparable.

## 4. Methods

### 4.1 Base models and training regimes

Implemented in `src/train.py`:

- Full fine-tuning (`--method full`)
- LoRA (`--method lora`, rank sweep supported)
- QLoRA (`--method qlora`)

Reproducibility improvements for final version:

- explicit seed control (`--seed`)
- per-run config snapshot (`run_config.json`)
- run naming includes method/model/lr/seed

### 4.2 Domain adaptation on NBA

Implemented in `src/train_nba.py`:

- starts from Spider-trained checkpoint
- continues training on NBA train split
- supports `--n-train` points for transfer curve (`10/20/70/all`)
- saves run config and seed metadata

### 4.3 Inference and evaluation

Implemented in `src/evaluate.py` with three schema modes:

1. Full schema context
2. Oracle tables (`tables_used`) upper bound
3. RAG retrieval (`top-k` tables via FAISS + sentence-transformers)

Metrics:

- Execution accuracy
- Exact match
- Difficulty-stratified breakdown

### 4.4 Prompt baselines

`src/prompt_baseline.py` supports:

- zero-shot baseline
- few-shot baseline (`--mode few_shot --few-shot-k`)
- NBA held-out split-compatible evaluation

### 4.5 Error analysis and reporting utilities

Added for final high-score workflow:

- `src/error_analysis.py`: rule-based taxonomy (`schema_linking`, `structural`, `value`, `aggregation`)
- `src/build_results_report.py`: generates summary tables and CSV exports for write-up/slides

## 5. Results

All headline numbers are from held-out NBA test split (`n=50`) using `eval/*_test.json`.

### 5.1 Main comparison (selected)

- `full_t5-base_nba_oracle_test`: exec 0.04, exact 0.00
- `lora_t5-base_r16_nba_oracle_test`: exec 0.04, exact 0.02
- `lora_codet5p-220m_r16_nba_oracle_test`: exec 0.10, exact 0.00
- `lora_codet5p-220m_r16_nba_nall_nba_oracle_test`: exec 0.46, exact 0.42

Observation: Spider-only checkpoints are weak in NBA domain, but adaptation with all NBA train examples yields a large jump for CodeT5+ LoRA.

### 5.2 Cross-domain adaptation trend

For CodeT5+ LoRA (oracle schema mode), performance rises with NBA adaptation data:

- n=20: exec 0.12 / exact 0.06
- n=70: exec 0.26 / exact 0.24
- n=all: exec 0.46 / exact 0.42

This indicates strong sample-efficiency gains but also clear headroom.

### 5.3 RAG ablation (current setup)

For `lora_codet5p-220m_r16_nba_nall`:

- oracle: exec 0.46
- RAG k=1: exec 0.12
- RAG k=3: exec 0.08
- RAG k=5: exec 0.08

In this configuration, RAG underperforms oracle schema by a wide margin, suggesting retrieval misses are a major bottleneck.

### 5.4 Spider controls (same 200-example slice)

Spider controls were refreshed to align the method comparison with current checkpoints:

- LoRA CodeT5 r16: exact 0.29
- Full T5-base: exact 0.265
- QLoRA T5-base: exact 0.14
- LoRA T5-base r16: exact 0.12
- LoRA T5-base r4: exact 0.07
- LoRA T5-base r8: exact 0.00

Across these runs, execution accuracy remains 0.00 on this quick 200-example protocol, so Spider-side comparison is currently driven by exact match rather than execution.

## 6. Analysis and Discussion

### 6.1 What worked

1. **Domain adaptation mattered more than source-only training.**  
   The jump from zero/low adaptation to n=all adaptation is much larger than differences among source-only methods.

2. **PEFT gave practical iteration speed.**  
   LoRA made it feasible to run many experiments under single-GPU constraints while retaining meaningful gains.

3. **Evaluation protocol is now reproducible.**  
   Fixed split, config snapshots, and result auto-aggregation reduce reporting inconsistency.

### 6.2 What did not work

1. **RAG quality is currently insufficient.**  
   Lower execution accuracy under RAG indicates retrieval quality and retrieval-to-prompt formatting need further improvement.

2. **Source-domain transfer gap remains large.**  
   Spider-trained models without NBA adaptation remain near-random on held-out NBA execution accuracy.

3. **Exact match is harsh for semantically equivalent SQL.**  
   Execution accuracy is more useful for practical performance, but still cannot distinguish partially correct structure.

### 6.3 Failure modes

Rule-based error taxonomy highlights four recurring classes:

- Schema linking (wrong table/column)
- Structural decoding errors (invalid/incomplete clauses)
- Value grounding errors (wrong literal values)
- Aggregation/grouping mistakes

These errors align with observed RAG underperformance and domain vocabulary mismatch.

### 6.4 Limitations

- Single-seed reporting for most runs (confidence intervals not yet included)
- NBA dataset is custom-authored; annotation noise may exist
- RAG retriever uses a simple dense setup without re-ranking

## 7. Conclusion

This project delivers a working applied text-to-SQL system and a training-focused experimental study under practical compute constraints. The strongest finding is that in-domain adaptation data is the dominant factor for NBA performance, while current retrieval quality limits RAG-based gains.

For next steps, priority should be:

1. improve retrieval quality and retrieval-conditioned prompting,
2. complete full method/rank sweeps with consistent seeds,
3. report confidence intervals and richer error analyses.

These additions can upgrade the project from a high-quality course deliverable to a stronger research submission candidate.

## References

- Yu et al. 2018, Spider
- Hu et al. 2021, LoRA
- Dettmers et al. 2023, QLoRA
- Lewis et al. 2020, RAG
- Raffel et al. 2020, T5
- Wang et al. 2021, CodeT5
