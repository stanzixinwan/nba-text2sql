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
- **NBA** (domain evaluation): SQLite DB built from [nba-sql](https://github.com/mpope9/nba-sql). ~10 tables (players, teams, games, player_game_log, etc.). Self-authored ~80–100 question/SQL pairs stratified by difficulty.

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

**Cross-domain curve:** Best Spider-trained model, evaluated on NBA with 0 / 10 / 50 / all NBA examples for few-shot adaptation.

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

## Running Experiments

```bash
# Baseline: zero-shot and few-shot
python src/prompt_baseline.py --model t5-base --mode zero_shot --eval spider
python src/prompt_baseline.py --model t5-base --mode few_shot --eval nba

# Full fine-tune
python src/train.py --model t5-base --method full --dataset spider --epochs 3

# LoRA with rank sweep
for rank in 4 8 16 32; do
    python src/train.py --model t5-base --method lora --rank $rank --dataset spider
done

# Evaluate best checkpoint with RAG
python src/evaluate.py --checkpoint models/lora_r16_best --use_rag --eval_set nba

# Launch demo
python src/demo.py --checkpoint models/lora_r16_best
```

## Metrics

- **Execution accuracy** — generated SQL returns the same result as gold SQL when executed
- **Exact match** — SQL string matches gold after canonicalization
- **Stratified accuracy** — split by Spider difficulty (easy/medium/hard/extra)
- **Error taxonomy** — schema linking errors, structural errors, value prediction errors, aggregation errors

## Deliverables

- [ ] Proposal (Apr 1)
- [ ] GitHub repository with reproducible scripts
- [ ] Write-up (≥4 pages PDF)
- [ ] Gradio demo
- [ ] Presentation slides (8–10 min)

## References

- Yu et al. 2018 — Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL
- Hu et al. 2021 — LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. 2023 — QLoRA: Efficient Finetuning of Quantized LLMs
- Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Raffel et al. 2020 — T5: Exploring the Limits of Transfer Learning
- Wang et al. 2021 — CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder