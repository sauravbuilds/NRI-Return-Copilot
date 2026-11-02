"""Human-review escalation rules as small pure functions."""
import config


def is_us_side_tax(scope_label: str) -> bool:
    return scope_label == "us_side_tax"


def has_complex_assets(persona) -> bool:
    n = len(set(persona.foreign_assets) & set(config.COMPLEX_ASSET_TYPES))
    return n >= config.COMPLEX_ASSET_COUNT_FOR_ESCALATION


def is_low_confidence(confidence: str) -> bool:
    return confidence == "low"


def community_contradicts(community_findings) -> bool:
    return any(f.verdict == "contradicts" for f in community_findings)


def escalation_reasons(persona, scope_label="in_scope", confidence="high",
                       community_findings=()) -> list[str]:
    reasons = []
    if is_us_side_tax(scope_label):
        reasons.append("US-side tax question: needs a US tax professional")
    if has_complex_assets(persona):
        reasons.append("Complex situation: several types of foreign assets")
    if is_low_confidence(confidence):
        reasons.append("Low confidence in the answer")
    if community_contradicts(community_findings):
        reasons.append("A community thread contradicts the official source")
    return reasons
