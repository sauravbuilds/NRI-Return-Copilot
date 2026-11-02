"""Print best-chunk relevance for answerable vs out-of-corpus questions.

Used to choose RELEVANCE_THRESHOLD from evidence.
These calibration questions are separate from the golden set.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.retrieval import retrieve  # noqa: E402

# Calibration questions. Each answerable one targets a document in the corpus.
ANSWERABLE = [
    "What happens to my NRO account when I return to India?",
    "Can an NRO account be redesignated as a resident rupee account?",
    "Can a resident individual hold a foreign currency account?",
    "What is Schedule FA in the income tax return?",
    "Which ITR forms include Schedule FA?",
    "Do I report foreign bank accounts in my return?",
    "What are CRS and FATCA?",
    "Which exchange rate is used to convert foreign assets into rupees in Schedule FA?",
]
OUT_OF_CORPUS = [
    "What is the best pizza topping?",
    "How do I renew a US passport?",
    "What is the capital gains tax rate on gold in Japan?",
    "Who won the cricket world cup?",
    "How do I apply for a UK student visa?",
]


def best_scores(questions):
    return [(q, max(c.score for c in retrieve(q))) for q in questions]


def main():
    for name, qs in (("ANSWERABLE", ANSWERABLE), ("OUT OF CORPUS", OUT_OF_CORPUS)):
        rows = best_scores(qs)
        print(f"\n{name}")
        for q, s in rows:
            print(f"  {s:6.3f}  {q}")
        vals = [s for _, s in rows]
        print(f"  min {min(vals):.3f}  mean {statistics.mean(vals):.3f}  max {max(vals):.3f}")


if __name__ == "__main__":
    main()
