import pytest

import config
from copilot.retrieval import Chunk, retrieve
from guardrails.grounding import check_grounding


def ch(score):
    return Chunk("t", {}, score)


def test_grounding_passes_above_threshold():
    assert check_grounding([ch(0.3), ch(0.9)], 0.5).passed


def test_grounding_abstains_below_threshold():
    assert not check_grounding([ch(0.2), ch(0.3)], 0.5).passed


def test_grounding_abstains_on_empty_retrieval():
    r = check_grounding([])
    assert not r.passed and "nothing" in r.reason


def test_grounding_uses_config_threshold_by_default():
    assert not check_grounding([ch(config.RELEVANCE_THRESHOLD - 0.01)]).passed
    assert check_grounding([ch(config.RELEVANCE_THRESHOLD)]).passed


@pytest.mark.parametrize("q,doc", [
    ("What happens to my NRO account when I return to India?", "rbi_nro_master_circular_2015"),
    ("What is Schedule FA in the income tax return?", "it_guide_fsi_tr_fa_2026"),
    ("What are CRS and FATCA?", "it_foreign_assets_transparency"),
])
def test_known_question_returns_expected_document_with_page_url(q, doc):
    chunks = retrieve(q)
    assert doc in {c.metadata["doc_id"] for c in chunks}
    top = next(c for c in chunks if c.metadata["doc_id"] == doc)
    assert top.metadata["page_url"].endswith(f"#page={top.metadata['page']}")
    assert top.metadata["issuer"] and top.metadata["last_verified"]


def test_irrelevant_question_scores_below_threshold():
    assert not check_grounding(retrieve("What is the best pizza topping?")).passed
