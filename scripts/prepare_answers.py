"""Prepare answers for the common questions in data/common_questions.json.

Run:  python scripts/prepare_answers.py

Each question goes through the normal pipeline, up to ATTEMPTS times. Only a full, sourced
answer with reasonable confidence is kept (see prepared.good_enough). The check for newer
publications is left out, because a prepared answer is reused for a long time and such a
notice would go stale. Run the script again after the documents are refreshed: answers
prepared from older documents are ignored by the app.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot import orchestrator, prepared  # noqa: E402

ATTEMPTS = 2


def prepare(question: str):
    """The best result in ATTEMPTS runs: the first one good enough, else the last."""
    result = None
    for _ in range(ATTEMPTS):
        result = orchestrator.run_pipeline(question, check_updates=False)
        if prepared.good_enough(result):
            break
        time.sleep(config.EVAL_SLEEP_SECONDS)
    return result


def main() -> None:
    questions = json.loads(prepared.QUESTIONS_PATH.read_text(encoding="utf-8"))
    entries = []
    for item in questions:
        result = prepare(item["question"])
        kept = prepared.good_enough(result)
        if kept:
            entries.append({"id": item["id"], "question": item["question"],
                            "answer": result.answer.model_dump(mode="json")})
        print(f"{item['id']}: {result.kind}{'' if kept else ' (not kept)'}")
        time.sleep(config.EVAL_SLEEP_SECONDS)
    prepared.save(entries)
    print(f"Prepared {len(entries)} of {len(questions)} answers")


if __name__ == "__main__":
    main()
