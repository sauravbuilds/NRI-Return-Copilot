"""Aggregate tagged threads with pandas and draw the pain-points chart.

Run:  python analytics/chart.py      -> analytics/pain_points.png
The chart always shows how many threads it is based on.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

TAGGED_PATH = config.ROOT / "analytics" / "threads_tagged.jsonl"
PNG_PATH = config.ROOT / "analytics" / "pain_points.png"
LABELS = {"residential_status": "Residential status", "nre_nro_accounts": "NRE / NRO accounts",
          "us_stocks_rsus": "US stocks / RSUs", "foreign_retirement": "Foreign retirement account",
          "foreign_asset_reporting": "Foreign asset reporting", "other": "Other"}
NOTE = ("Sample: public threads found by search. This is a biased sample (people who post online "
        "and what the search engine surfaces), not a measure of all returning NRIs.")


def load(path=TAGGED_PATH) -> pd.DataFrame:
    p = Path(path)
    return pd.read_json(p, lines=True) if p.exists() and p.stat().st_size else pd.DataFrame()


def aggregate(df: pd.DataFrame) -> dict:
    """Counts by problem type, stage and resolved, plus the sample size."""
    if df.empty:
        return {"n": 0, "problem_type": pd.Series(dtype=int), "stage": pd.Series(dtype=int),
                "resolved": pd.Series(dtype=int)}
    return {"n": len(df),
            "problem_type": df["problem_type"].value_counts(),
            "stage": df["stage"].value_counts(),
            "resolved": df["resolved"].value_counts()}


def make_chart(df: pd.DataFrame, out=PNG_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = aggregate(df)
    counts = agg["problem_type"].rename(index=LABELS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("Number of threads")
    ax.set_title(f"Most common pain points for returning NRIs (n = {agg['n']} threads)")
    fig.text(0.01, 0.01, NOTE, fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


if __name__ == "__main__":
    data = load()
    if data.empty:
        sys.exit("No tagged threads yet. Run analytics/collect.py then analytics/tag.py.")
    print("saved", make_chart(data))
