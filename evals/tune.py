"""Tuning runs: every chunk size x every abstention threshold, logged to evals/TUNING_LOG.md.

Run:  python evals/tune.py        (no model key needed)
Each chunk size gets its own temporary index, so the real index is never touched and one
size cannot reuse another's cached data. The golden set's answerable and out-of-corpus
questions then go through the document search and the grounding check. Chunk size changes
what is retrieved and the threshold changes what is answered, and both show up at this
stage, so the grid needs no model calls and can be repeated freely.
The observation column is filled in by hand afterwards from the real numbers.
"""
import gc
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from copilot import retrieval  # noqa: E402
from evals import run_evals  # noqa: E402
from scripts.ingest import build_index  # noqa: E402

LOG = config.EVALS_DIR / "TUNING_LOG.md"


def fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def main() -> None:
    settings = (config.CHROMA_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.RELEVANCE_THRESHOLD)
    rows, counts, n_ans, n_ooc = [], {}, 0, 0
    try:
        for size in config.TUNING_CHUNK_SIZES:
            config.CHROMA_DIR = settings[0].parent / f"chroma_tune_{size}"
            config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, int(size * config.CHUNK_OVERLAP_RATIO)
            retrieval._store = None
            counts[size] = build_index(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            for threshold in config.TUNING_THRESHOLDS:
                config.RELEVANCE_THRESHOLD = threshold
                m = run_evals.retrieval_stage_metrics()
                n_ans, n_ooc = m["n_answerable"], m["n_out_of_corpus"]
                rows.append(f"| chunk {size}, threshold {threshold:.2f} | {fmt(m['retrieval_hit_rate'])} | "
                            f"{fmt(m['answerable_passing'])} | {fmt(m['out_of_corpus_abstained'])} | (fill in) |")
            retrieval._store = None
            gc.collect()
            shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    finally:
        config.CHROMA_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.RELEVANCE_THRESHOLD = settings
        retrieval._store = None
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {date.today()}: tuning grid (chunk size x threshold)\n\n")
        f.write(f"Measured at the retrieval stage on {n_ans} answerable and {n_ooc} out-of-corpus golden questions. "
                f"Chunks indexed: {', '.join(f'{n} at size {s}' for s, n in counts.items())}.\n\n")
        f.write("| Config | Retrieval hit rate | Answerable questions passing the threshold | "
                "Out-of-corpus questions abstained | Observation |\n")
        f.write("|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
