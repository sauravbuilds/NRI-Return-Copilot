import pytest
from pydantic import ValidationError

from copilot.schemas import Answer, CommunityFinding, Citation, Finding, Persona, RoadmapStep


def cite(page=3):
    return Citation(document_title="T", issuer="RBI", page=page,
                    page_url=f"https://x.org/a.pdf#page={page}", last_verified="2026-01-01")


def test_valid_objects_pass():
    Finding(statement="s", citations=[cite()])
    RoadmapStep(stage="first_30_days", step="s", citations=[cite()])
    assert Answer(summary="x").disclaimer
    assert Persona().concerns == ""


def test_uncited_finding_rejected():
    with pytest.raises(ValidationError):
        Finding(statement="s", citations=[])


def test_uncited_roadmap_step_rejected():
    with pytest.raises(ValidationError):
        RoadmapStep(stage="first_year", step="s", citations=[])


def test_invalid_verdict_rejected():
    with pytest.raises(ValidationError):
        CommunityFinding(url="u", title="t", snippet="s", verdict="maybe", reason="r")


def test_page_url_must_match_page():
    with pytest.raises(ValidationError):
        Citation(document_title="T", issuer="RBI", page=3,
                 page_url="https://x.org/a.pdf#page=4", last_verified="2026-01-01")


def test_invalid_relocation_status_rejected():
    with pytest.raises(ValidationError):
        Persona(relocation_status="someday")
