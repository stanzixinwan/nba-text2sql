# NBA Text-to-SQL with PEFT & RAG

End-to-end text-to-SQL system that translates natural language questions into executable SQL. Trained on Spider, evaluated on a custom NBA statistics database. Systematic comparison of full fine-tuning vs. LoRA vs. QLoRA, with a RAG pipeline for schema retrieval.

**Course:** COSI 115b Fundamentals of NLP II | **Deadline:** May 6, 2026

---

## Goals

1. Build a working text-to-SQL pipeline (data → train → eval → demo)
2. Systematically compare training regimes: prompt engineering, full fine-tuning, LoRA, QLoRA
3. Study cross-domain generalization from Spider (general) → NBA (domain)
4. Quantify RAG's contribution to schema linking accuracy
5. Ship a Gradio demo answering natural language NBA questions

## Tech Stack

PyTorch · HuggingFace Transformers · PEFT (LoRA/QLoRA) · Datasets · Accelerate · sentence-transformers · FAISS · Gradio · SQLite · sqlparse

## Datasets

- **Spider** (training): `xlangai/spider` on HuggingFace. 10,181 questions, 200 databases, 138 domains.
- **NBA** (domain evaluation): SQLite DB built from [nba-sql](https://github.com/mpope9/nba-sql). ~10 tables (players, teams, games, player_game_log, etc.). Self-authored ~200 question/SQL pairs stratified by difficulty.

## Models

- `t5-base` (primary)
- `Salesforce/codet5-base` (comparison — code-pretrained)

## Project Structure

```
nba-text2sql/
├── data/
│   ├── raw/              # Spider download, NBA sqlite
│   ├── processed/        # Unified train/dev/test JSON
│   └── nba/              # Self-authored NBA question/SQL pairs
├── src/
│   ├── data_utils.py     # Loading, schema serialization, input formatting
│   ├── rag.py            # sentence-transformer + FAISS schema retrieval
│   ├── train.py          # Full fine-tune + LoRA/QLoRA training loop
│   ├── evaluate.py       # Execution accuracy, exact match, error taxonomy
│   ├── prompt_baseline.py # Zero-shot & few-shot baselines
│   └── demo.py           # Gradio interface
├── notebooks/            # Exploratory analysis, result plots
├── models/               # Saved checkpoints (gitignored)
├── eval/                 # Result JSONs, confusion matrices, plots
├── requirements.txt
└── README.md
```

## Experimental Matrix

| Condition                | Spider dev | NBA eval |
|--------------------------|:----------:|:--------:|
| Zero-shot T5             |     ✓      |    ✓     |
| Few-shot T5              |     ✓      |    ✓     |
| Full fine-tune T5        |     ✓      |    ✓     |
| LoRA T5 (rank 4/8/16/32) |     ✓      |    ✓     |
| QLoRA T5                 |     ✓      |    ✓     |
| LoRA CodeT5              |     ✓      |    ✓     |
| Best model + RAG         |     ✓      |    ✓     |
| Best model + RAG + few-shot NBA | —     |    ✓     |

**Cross-domain curve:** Best Spider-trained model, evaluated on NBA with 0 / 20 / 70 / all NBA examples for few-shot adaptation.

## Timeline

| Week | Dates | Milestones |
|------|-------|------------|
| 1 | Apr 1–7 | Repo setup · Spider + NBA data loaded · NBA Q/SQL pairs authored · zero/few-shot baselines working |
| 2 | Apr 8–14 | Full fine-tune T5-base on Spider · RAG pipeline built · RAG on/off comparison |
| 3 | Apr 15–21 | LoRA/QLoRA ablations (rank, LR, modules) · T5 vs CodeT5 · few-shot NBA adaptation curve |
| 4 | Apr 22–28 | NBA evaluation + error taxonomy · Gradio demo · start write-up with all tables/figures |
| 5 | Apr 29–May 6 | Polish write-up · clean repo · slides · present May 6 |

## Setup

```bash
git clone <repo>
cd nba-text2sql
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download Spider via HuggingFace (automatic on first use)
python src/data_utils.py --download spider

# Build NBA SQLite database (takes ~30 min for current season, hours for full history)
git clone https://github.com/mpope9/nba-sql.git
cd nba-sql && python stats/nba_sql.py --database sqlite --sqlite-path ../data/raw/nba.sqlite
```

### GPU (Windows, NVIDIA)

Use **Python 3.12 or 3.13** with **PyTorch CUDA** — Python **3.14** does not ship official CUDA wheels yet, so `pip install torch` often resolves to **CPU-only**.

```powershell
.\scripts\setup_gpu_venv.ps1
.\.venv\Scripts\Activate.ps1
```

Details: [GPU_SETUP.md](GPU_SETUP.md). Training (`train.py`, `train_nba.py`) and NBA evaluation (`evaluate.py`) **exit unless CUDA is available**, unless you set `NBA_TEXT2SQL_ALLOW_CPU=1` (debug only).


### One-command command generation

- Generate all commands for a stage without running:
  - `python -m src.run_experiment_matrix --stage rank_sweep --dry-run`
  - `python -m src.run_experiment_matrix --stage method_compare --dry-run`
  - `python -m src.run_experiment_matrix --stage adaptation --dry-run`
- Execute by swapping `--dry-run` to `--run`.

## Metrics

- **Execution accuracy** — generated SQL returns the same result as gold SQL when executed
- **Exact match** — SQL string matches gold after canonicalization
- **Stratified accuracy** — split by Spider difficulty (easy/medium/hard/extra)
- **Error taxonomy** — schema linking errors, structural errors, value prediction errors, aggregation errors

## Deliverables

- [x] Proposal (Apr 1)
- [x] GitHub repository with reproducible scripts
- [x] NBA held-out test split evaluation protocol
- [x] Gradio demo (`src/demo.py`)
- [x] Write-up (>=4 pages PDF)
- [x] Presentation slides (8–10 min)

## Reproducibility Notes

- All comparisons should use `--split test` for NBA to avoid train-set leakage.
- `src/train.py` and `src/train_nba.py` now save `run_config.json` per run.
- Run names include learning rate and seed for traceability.
- Aggregated tables for paper/slides are auto-generated by `src/build_results_report.py`.
- RAG: `python -m src.rag --build` (dense FAISS) and `python -m src.rag --build-bm25`; evaluate with `--use-rag --rag-backend dense|bm25|hybrid --top-k K`. Full retrieval + NBA test sweep: `python scripts/run_rag_retrieval_ablation.py` (or `python scripts/run_rag_retrieval_ablation.py --aggregate-only` to rebuild CSVs from `eval/*_nba_rag_*_test.json`). Figures: `python scripts/plot_rag_ablation.py`.

## References

- Yu et al. 2018 — Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL
- Hu et al. 2021 — LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. 2023 — QLoRA: Efficient Finetuning of Quantized LLMs
- Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Raffel et al. 2020 — T5: Exploring the Limits of Transfer Learning
- Wang et al. 2021 — CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder
