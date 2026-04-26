"""
data_utils.py — Unified data pipeline for Spider + NBA text-to-SQL.

Core responsibilities:
1. Serialize database schemas into text strings for model input
2. Format (question, schema) → input_text and SQL → target_text
3. Provide a unified loader so training/eval code doesn't care about data source

Usage:
    from src.data_utils import load_spider_splits, load_nba_dataset, format_example

    # Spider
    train, dev = load_spider_splits()

    # NBA
    nba = load_nba_dataset("data/nba/nba_questions.json", "data/raw/nba.sqlite")

    # Both return list[dict] with keys: input_text, target_text, db_id, difficulty
    print(train[0]["input_text"])
    print(train[0]["target_text"])
"""

import json
import sqlite3
from typing import Optional


# ──────────────────────────────────────────────
# 1. Schema serialization
# ──────────────────────────────────────────────

def get_sqlite_schema(db_path: str, tables: Optional[list[str]] = None) -> dict:
    """
    Extract schema from a SQLite database.

    Returns:
        {
            "table_name": [
                {"column": "col_name", "type": "TEXT"},
                ...
            ],
            ...
        }
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if tables is None:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]

    schema = {}
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [{"column": row[1], "type": row[2]} for row in cur.fetchall()]
        if cols:
            schema[table] = cols

    conn.close()
    return schema


def serialize_schema(schema: dict, include_types: bool = False) -> str:
    """
    Convert schema dict to a flat text string for model input.

    Example output (include_types=False):
        "team : id , full_name , abbreviation , city , state , year_founded | game : season_id , ..."

    Example output (include_types=True):
        "team : id (TEXT) , full_name (TEXT) | game : season_id (TEXT) , ..."
    """
    parts = []
    for table_name, columns in schema.items():
        if include_types:
            col_strs = [f"{c['column']} ({c['type']})" for c in columns]
        else:
            col_strs = [c["column"] for c in columns]
        parts.append(f"{table_name} : {' , '.join(col_strs)}")
    return " | ".join(parts)


def serialize_schema_for_table(schema: dict, table_name: str, include_types: bool = False) -> str:
    """Serialize schema for a single table."""
    if table_name not in schema:
        return ""
    columns = schema[table_name]
    if include_types:
        col_strs = [f"{c['column']} ({c['type']})" for c in columns]
    else:
        col_strs = [c["column"] for c in columns]
    return f"{table_name} : {' , '.join(col_strs)}"


# ──────────────────────────────────────────────
# 2. Input/output formatting
# ──────────────────────────────────────────────

def format_input(question: str, schema_text: str) -> str:
    """
    Create the model input string.

    Format:
        "translate to SQL: {question} | {schema_text}"

    This prefix style works well with T5/CodeT5 seq2seq models.
    """
    return f"translate to SQL: {question} | {schema_text}"


def format_target(sql: str) -> str:
    """
    Normalize the target SQL string.
    - Strip whitespace
    - Remove trailing semicolons (T5 tokenizer handles this better without them)
    """
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def format_example(question: str, schema: dict, sql: str,
                   include_types: bool = False) -> dict:
    """
    Format a single (question, schema, sql) triple into model-ready strings.

    Returns:
        {"input_text": "translate to SQL: ...", "target_text": "SELECT ..."}
    """
    schema_text = serialize_schema(schema, include_types=include_types)
    return {
        "input_text": format_input(question, schema_text),
        "target_text": format_target(sql),
    }


# ──────────────────────────────────────────────
# 3. Spider loader
# ──────────────────────────────────────────────

SPIDER_DATASET_NAME = "SuperMax991/spider-text2sql"


def _schema_from_spider_tables(table_examples: list[dict]) -> dict:
    """Convert Spider tables.json records into this module's schema format."""
    schemas = {}
    for ex in table_examples:
        table_names = ex["table_names_original"]
        column_types = ex.get("column_types", [])
        schema = {t: [] for t in table_names}

        for col_idx, (table_idx, col_name) in enumerate(ex["column_names_original"]):
            if table_idx < 0:
                continue
            table_name = table_names[table_idx]
            col_type = column_types[col_idx] if col_idx < len(column_types) else ""
            schema[table_name].append({"column": col_name, "type": col_type})

        schemas[ex["db_id"]] = schema
    return schemas


def _load_spider_schemas(spider_dataset) -> dict:
    """
    Extract per-database schemas from Spider's HuggingFace dataset.

    Spider stores schema info in each example. We deduplicate by db_id.
    Returns: {db_id: schema_dict}
    """
    first_split = next(iter(spider_dataset))
    first_example = spider_dataset[first_split][0]

    if "table_names_original" not in first_example:
        raise ValueError(
            "Spider dataset does not include schema metadata. Use a dataset with "
            "db_schema or table_names_original fields."
        )

    schemas = {}
    for split in spider_dataset:
        for ex in spider_dataset[split]:
            db_id = ex["db_id"]
            if db_id in schemas:
                continue

            schemas[db_id] = _schema_from_spider_tables([ex])[db_id]
    return schemas


def load_spider_splits(include_types: bool = False, max_examples: Optional[int] = None):
    """
    Load Spider train and validation splits, formatted for model input.

    Returns:
        train_data: list[dict]  — each has input_text, target_text, db_id, difficulty
        dev_data:   list[dict]

    Each dict has keys:
        - input_text: str
        - target_text: str
        - db_id: str
        - difficulty: str (Spider's difficulty label)
    """
    from datasets import load_dataset
    spider = load_dataset(SPIDER_DATASET_NAME)
    has_db_schema = "db_schema" in spider["train"].column_names
    schemas = {} if has_db_schema else _load_spider_schemas(spider)

    def process_split(split_name):
        data = []
        for ex in spider[split_name]:
            db_id = ex["db_id"]
            if "db_schema" in ex:
                formatted = {
                    "input_text": format_input(ex["question"], ex["db_schema"]),
                    "target_text": format_target(ex["query"]),
                }
            else:
                schema = schemas.get(db_id, {})
                formatted = format_example(
                    question=ex["question"],
                    schema=schema,
                    sql=ex["query"],
                    include_types=include_types,
                )
            formatted["db_id"] = db_id
            formatted["difficulty"] = ex.get("difficulty", "unknown")
            data.append(formatted)
            if max_examples and len(data) >= max_examples:
                break
        return data

    train_data = process_split("train")
    dev_split = "validation" if "validation" in spider else "test"
    dev_data = process_split(dev_split)
    return train_data, dev_data


# ──────────────────────────────────────────────
# 4. NBA loader
# ──────────────────────────────────────────────

# Tables to include in the NBA schema for model input.
# Excluding play_by_play (too many columns, rarely needed for most queries)
# and team_info_common (empty table).
NBA_CORE_TABLES = ["team", "player", "common_player_info", "game", "draft_history"]


def load_nba_dataset(
    questions_path: str = "data/nba/nba_questions.json",
    db_path: str = "data/raw/nba.sqlite",
    include_types: bool = False,
    tables: Optional[list[str]] = None,
) -> list[dict]:
    """
    Load NBA question/SQL pairs, formatted for model input.

    Returns list[dict] with keys:
        - input_text: str
        - target_text: str
        - db_id: "nba"
        - difficulty: str
        - tables_used: list[str]
        - sql_components: list[str]
        - question: str  (original question, useful for error analysis)
        - gold_sql: str  (original SQL, useful for execution accuracy eval)
    """
    tables = tables or NBA_CORE_TABLES
    schema = get_sqlite_schema(db_path, tables=tables)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    data = []
    for q in questions:
        formatted = format_example(
            question=q["question"],
            schema=schema,
            sql=q["query"],
            include_types=include_types,
        )
        formatted["db_id"] = "nba"
        formatted["difficulty"] = q.get("difficulty", "unknown")
        formatted["tables_used"] = q.get("tables_used", [])
        formatted["sql_components"] = q.get("sql_components", [])
        formatted["question"] = q["question"]
        formatted["gold_sql"] = q["query"]
        data.append(formatted)

    return data


# ──────────────────────────────────────────────
# 5. NBA schema document (for RAG embedding)
# ──────────────────────────────────────────────

def build_nba_schema_documents(
    db_path: str = "data/raw/nba.sqlite",
    tables: Optional[list[str]] = None,
) -> list[dict]:
    """
    Build per-table schema documents for RAG retrieval.

    Each document is a dict:
        {"table": "game", "text": "game : season_id , team_id_home , ..."}

    These documents will be embedded with sentence-transformers and stored in FAISS.
    At inference time, the user's question is encoded, top-k tables are retrieved,
    and only those tables' schemas are injected into the prompt.
    """
    tables = tables or NBA_CORE_TABLES
    schema = get_sqlite_schema(db_path, tables=tables)

    documents = []
    for table_name in schema:
        text = serialize_schema_for_table(schema, table_name, include_types=True)
        documents.append({"table": table_name, "text": text})

    return documents


# ──────────────────────────────────────────────
# 6. CLI: quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-nba", action="store_true", help="Test NBA data loading")
    parser.add_argument("--test-spider", action="store_true", help="Test Spider data loading")
    parser.add_argument("--nba-questions", default="data/nba/nba_questions.json")
    parser.add_argument("--nba-db", default="data/raw/nba.sqlite")
    args = parser.parse_args()

    if args.test_nba:
        print("Loading NBA dataset...")
        nba = load_nba_dataset(args.nba_questions, args.nba_db)
        print(f"  Loaded {len(nba)} examples")
        print(f"\n  Sample input (first 200 chars):\n    {nba[0]['input_text'][:200]}...")
        print(f"\n  Sample target:\n    {nba[0]['target_text']}")

        print("\n  NBA schema documents for RAG:")
        docs = build_nba_schema_documents(args.nba_db)
        for d in docs[:3]:
            print(f"    [{d['table']}] {d['text'][:80]}...")

    if args.test_spider:
        print("Loading Spider dataset (this downloads ~100MB on first run)...")
        train, dev = load_spider_splits(max_examples=5)
        print(f"  Train: {len(train)} examples (showing first 5)")
        print(f"  Dev:   {len(dev)} examples (showing first 5)")
        print(f"\n  Sample input (first 200 chars):\n    {train[0]['input_text'][:200]}...")
        print(f"\n  Sample target:\n    {train[0]['target_text']}")