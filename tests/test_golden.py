import json
from pathlib import Path

GOLDEN = Path(__file__).parent.parent / "evals" / "golden.jsonl"
CORPUS = Path(__file__).parent.parent / "corpus"

VALID_OUTCOMES = {"auto_resolve", "escape_hatch", "escalate"}
VALID_DIFFICULTY = {"easy", "near_miss", "unanswerable"}


def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_file_is_valid_json():
    assert len(load_golden())>0


def test_ids_are_unique():
    ids=[row["id"] for row in load_golden()]
    assert len(ids)==len(set(ids))


def test_allowed_values():
    for row in load_golden():
        assert row["expected_outcome"] in VALID_OUTCOMES, row["id"]
        assert row["difficulty"] in VALID_DIFFICULTY, row["id"]

def test_supporting_docs_exist():
    doc_ids={p.stem for p in CORPUS.glob("*.md")}
    for row in load_golden():
        for doc_id in row["supporting_doc_ids"]:
            assert doc_id in doc_ids, f"{row['id']}: {doc_id}"