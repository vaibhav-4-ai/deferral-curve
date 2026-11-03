import json
from pathlib import Path

from dc.agent import Agent
from dc.corpus import load_corpus
from dc.eval_retrieval import load_golden
from dc.grade import is_correct
from dc.llm import default_llm
from dc.retrieve import HybridRetriever

RESULTS = Path(__file__).parent.parent / "evals" / "results.jsonl"


def run():
    # Runs the agent on every golden question and saves one line per question: the signals it
    # produced and whether its answer was right. Needs GROQ_API_KEY the first time; later runs replay the cache.
    agent = Agent(HybridRetriever(load_corpus()), default_llm(pace=2.5))
    rows = []
    for item in load_golden():
        result = agent.answer(item["query"])
        correct = is_correct(item, result)
        rows.append(
            {
                "id": item["id"],
                "difficulty": item["difficulty"],
                "expected_outcome": item["expected_outcome"],
                "correct": correct,
                "verified": result.verified,
                "degraded": result.degraded,
                "tool_used": result.tool_used,
                "llm_calls": result.llm_calls,
                "features": result.features,
                "answer": result.answer,
            }
        )
        print(f"{item['id']} correct={correct} verified={result.verified} degraded={result.degraded} {result.error}")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    answered = [r for r in rows if not r["degraded"]]
    right = sum(r["correct"] for r in answered)
    print(f"{right}/{len(answered)} correct when the model was available ({len(rows) - len(answered)} degraded)")
