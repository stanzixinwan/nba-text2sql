"""
rag.py — RAG-based schema retrieval for NBA text-to-SQL.

Replaces oracle table selection with real retrieval:
1. Encode each NBA table's schema description with sentence-transformers
2. Index in FAISS
3. At inference, encode the question and retrieve top-k tables
4. Use only retrieved tables' schemas in the prompt

Usage:
    # Build index (one-time)
    python -m src.rag --build

    # Eval retrieval quality (does retrieved tables ⊇ gold tables_used?)
    python -m src.rag --eval-retrieval --top-k 3

    # Use in evaluation:
    from src.rag import SchemaRetriever
    retriever = SchemaRetriever()
    retriever.load()
    tables = retriever.retrieve("How many teams are in the NBA?", top_k=3)
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.data_utils import (
    build_nba_schema_documents,
    get_sqlite_schema,
    serialize_schema,
    format_input,
    format_target,
    NBA_CORE_TABLES,
)


INDEX_DIR = Path("models/rag_index")


class SchemaRetriever:
    """sentence-transformer + FAISS retriever for NBA table schemas."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents = None

    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)

    def build(self, db_path: str = "data/raw/nba.sqlite",
              tables: Optional[list[str]] = None,
              save_dir: Path = INDEX_DIR):
        """Build FAISS index from NBA schema documents."""
        import faiss

        self._load_encoder()
        self.documents = build_nba_schema_documents(db_path, tables)
        texts = [d["text"] for d in self.documents]

        print(f"Encoding {len(texts)} table schemas with {self.model_name}...")
        embeddings = self.encoder.encode(texts, normalize_embeddings=True,
                                          show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "schema.faiss"))
        with open(save_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Index built: {len(self.documents)} tables, dim={embeddings.shape[1]}")
        print(f"Saved to {save_dir}/")

    def load(self, save_dir: Path = INDEX_DIR):
        """Load a previously built index."""
        import faiss
        self._load_encoder()
        self.index = faiss.read_index(str(save_dir / "schema.faiss"))
        with open(save_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        """Return top-k table names most relevant to the question."""
        if self.index is None:
            raise RuntimeError("Index not loaded. Call .load() or .build() first.")
        query_emb = self.encoder.encode([question], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)
        return [self.documents[i]["table"] for i in indices[0]]

    def retrieve_with_scores(self, question: str, top_k: int = 3) -> list[tuple]:
        """Return (table_name, score) pairs."""
        query_emb = self.encoder.encode([question], normalize_embeddings=True)
        query_emb = np.array(query_emb).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)
        return [(self.documents[i]["table"], float(scores[0][j]))
                for j, i in enumerate(indices[0])]


# ──────────────────────────────────────────────
# RAG-aware NBA dataset loader
# ──────────────────────────────────────────────

def load_nba_dataset_with_rag(
    questions_path: str = "data/nba/nba_questions.json",
    db_path: str = "data/raw/nba.sqlite",
    retriever: Optional[SchemaRetriever] = None,
    top_k: int = 3,
    include_types: bool = True,
) -> list[dict]:
    """
    Load NBA examples with RAG-retrieved schemas (instead of full or oracle).

    Each example's prompt only contains schemas of the top-k retrieved tables.
    Useful for end-to-end RAG evaluation.

    Returns list[dict] with same keys as load_nba_dataset, plus:
        - retrieved_tables: list[str]  — what RAG selected
        - retrieval_recall: float      — fraction of gold tables retrieved
    """
    if retriever is None:
        retriever = SchemaRetriever()
        retriever.load()

    full_schema = get_sqlite_schema(db_path, tables=NBA_CORE_TABLES)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    data = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        # Build schema from retrieved tables only
        schema = {t: full_schema[t] for t in retrieved if t in full_schema}

        # Compute retrieval recall (how many gold tables did we retrieve?)
        gold_tables = set(q.get("tables_used", []))
        retrieved_set = set(retrieved)
        if gold_tables:
            recall = len(gold_tables & retrieved_set) / len(gold_tables)
        else:
            recall = 1.0

        schema_text = serialize_schema(schema, include_types=include_types)
        data.append({
            "input_text": format_input(q["question"], schema_text),
            "target_text": format_target(q["query"]),
            "db_id": "nba",
            "difficulty": q.get("difficulty", "unknown"),
            "tables_used": q.get("tables_used", []),
            "sql_components": q.get("sql_components", []),
            "question": q["question"],
            "gold_sql": q["query"],
            "retrieved_tables": retrieved,
            "retrieval_recall": recall,
        })

    return data


# ──────────────────────────────────────────────
# Retrieval-only evaluation (no SQL generation)
# ──────────────────────────────────────────────

def evaluate_retrieval(questions_path: str, top_k: int = 3,
                       retriever: Optional[SchemaRetriever] = None) -> dict:
    """
    Measure retrieval quality alone (no LLM needed).

    Reports:
      - Recall@k: fraction of gold tables among retrieved
      - Perfect-recall rate: fraction of examples where ALL gold tables retrieved
      - Per-difficulty breakdown
    """
    from collections import defaultdict

    if retriever is None:
        retriever = SchemaRetriever()
        retriever.load()

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    by_diff = defaultdict(lambda: {"total": 0, "recall_sum": 0.0, "perfect": 0})

    for q in questions:
        retrieved = set(retriever.retrieve(q["question"], top_k=top_k))
        gold = set(q.get("tables_used", []))
        if not gold:
            continue
        recall = len(gold & retrieved) / len(gold)
        perfect = int(gold.issubset(retrieved))
        diff = q.get("difficulty", "unknown")
        by_diff[diff]["total"] += 1
        by_diff[diff]["recall_sum"] += recall
        by_diff[diff]["perfect"] += perfect

    total = sum(d["total"] for d in by_diff.values())
    overall_recall = sum(d["recall_sum"] for d in by_diff.values()) / total
    overall_perfect = sum(d["perfect"] for d in by_diff.values()) / total

    print(f"\n=== Retrieval @ top-{top_k} ({total} examples) ===")
    print(f"  Avg recall:        {overall_recall:.1%}")
    print(f"  Perfect retrieval: {overall_perfect:.1%}  (all gold tables in top-{top_k})")
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard", "extra_hard"]:
        if diff in by_diff and by_diff[diff]["total"] > 0:
            d = by_diff[diff]
            print(f"    {diff:11s}: recall {d['recall_sum']/d['total']:.1%} | "
                  f"perfect {d['perfect']}/{d['total']} ({d['perfect']/d['total']:.1%})")

    return {"overall_recall": overall_recall, "perfect_rate": overall_perfect}


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true",
                        help="Build FAISS index from NBA schema")
    parser.add_argument("--eval-retrieval", action="store_true",
                        help="Evaluate retrieval quality (no LLM)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo retrieval on a few sample questions")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--db-path", default="data/raw/nba.sqlite")
    parser.add_argument("--questions", default="data/nba/nba_questions.json")
    args = parser.parse_args()

    if args.build:
        retriever = SchemaRetriever()
        retriever.build(db_path=args.db_path)

    if args.eval_retrieval:
        evaluate_retrieval(args.questions, top_k=args.top_k)

    if args.demo:
        retriever = SchemaRetriever()
        retriever.load()
        samples = [
            "How many teams are in the NBA?",
            "Which team had the highest 3-point shooting percentage at home in playoffs?",
            "List players drafted in the first round of 2018.",
            "What arena does the team_details list for the Lakers?",
        ]
        for q in samples:
            results = retriever.retrieve_with_scores(q, top_k=args.top_k)
            print(f"\nQ: {q}")
            for table, score in results:
                print(f"   [{score:.3f}] {table}")