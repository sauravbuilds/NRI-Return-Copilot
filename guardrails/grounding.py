"""Grounding guard: abstain when no official text supports an answer."""
from dataclasses import dataclass

import config


@dataclass
class GroundingResult:
    passed: bool
    reason: str


def check_grounding(chunks, threshold: float | None = None) -> GroundingResult:
    threshold = config.RELEVANCE_THRESHOLD if threshold is None else threshold
    if not chunks:
        return GroundingResult(False, "retrieval returned nothing")
    best = max(c.score for c in chunks)
    if best < threshold:
        return GroundingResult(False, f"best relevance {best:.2f} is below threshold {threshold:.2f}")
    return GroundingResult(True, "supported")
