"""Data models. Pydantic rejects bad data, so rules like "every step must
be cited" are enforced by the type itself, not by hoping the LLM behaves."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

DISCLAIMER = (
    "Quick note: this is general information from official documents, not legal, "
    "tax or investment advice. Please check with a qualified chartered "
    "accountant before you act on it."
)

RelocationStatus = Literal["planning", "relocated_recently", "relocated_over_a_year_ago", "undecided"]
IndianAccount = Literal["NRE", "NRO", "resident_savings", "none"]
ForeignAsset = Literal["us_stocks", "vested_rsus_esops", "mutual_funds",
                       "foreign_retirement_account", "property", "none"]
Goal = Literal["move_assets_to_india", "keep_abroad", "understand_reporting"]
Verdict = Literal["agrees", "contradicts", "unrelated"]
Confidence = Literal["low", "medium", "high"]
Stage = Literal["before_you_move", "first_30_days", "first_year"]


class DaysInIndia(BaseModel):
    year: int
    days: int = Field(ge=0, le=366)


class Persona(BaseModel):
    """Context only. No names, numbers or documents. Unknown stays empty."""
    relocation_status: Optional[RelocationStatus] = None
    move_date: Optional[date] = None
    countries_lived: list[str] = []
    days_in_india_by_year: list[DaysInIndia] = []
    indian_accounts: list[IndianAccount] = []
    foreign_assets: list[ForeignAsset] = []
    goals: list[Goal] = []
    concerns: str = ""


class Citation(BaseModel):
    document_title: str
    issuer: str
    page: int = Field(ge=1)
    page_url: str  # document url + "#page=N"
    last_verified: str
    doc_id: str = ""
    published: str = ""  # release date of the document (ISO), empty when unknown

    def model_post_init(self, _ctx) -> None:
        if not self.page_url.endswith(f"#page={self.page}"):
            raise ValueError("page_url must end with #page=<page>")

    @property
    def document_url(self) -> str:
        return self.page_url.rsplit("#", 1)[0]


class Finding(BaseModel):
    statement: str
    citations: list[Citation] = Field(min_length=1)  # uncited = cannot exist


class RoadmapStep(BaseModel):
    stage: Stage
    step: str
    citations: list[Citation] = Field(min_length=1)


class MissingItem(BaseModel):
    field: str
    why_it_matters: str
    if_this_then_that: str


class CommunityFinding(BaseModel):
    url: str
    title: str
    snippet: str
    verdict: Verdict
    reason: str


class OfficialUpdate(BaseModel):
    """An official publication newer than the documents the answer is based on.
    'unchecked' means the comparison could not run."""
    url: str
    title: str
    snippet: str
    date: str = ""
    verdict: Literal["agrees", "changes", "unrelated", "unchecked"]
    reason: str


class CABrief(BaseModel):
    situation: str
    open_questions: list[str]
    pages_to_read: list[Citation]


class Answer(BaseModel):
    summary: str
    official_findings: list[Finding] = []
    roadmap: list[RoadmapStep] = []
    missing_information: list[MissingItem] = []
    community_findings: list[CommunityFinding] = []
    official_updates: list[OfficialUpdate] = []
    confidence: Confidence = "low"
    confidence_reason: str = ""
    needs_human_review: bool = False
    review_reasons: list[str] = []
    ca_brief: Optional[str] = None
    abstained: bool = False
    disclaimer: str = DISCLAIMER


# --- Structured outputs the LLM fills in (the app turns these into Answer) ----
class ScopeResult(BaseModel):
    label: Literal["in_scope", "investment_advice", "us_side_tax", "out_of_scope"]
    reason: str = ""


class VerdictResult(BaseModel):
    label: Verdict
    reason: str


class SearchPlan(BaseModel):
    community_query: str
    official_query: str


class UpdateVerdictResult(BaseModel):
    label: Literal["agrees", "changes", "unrelated"]
    reason: str


class DraftFinding(BaseModel):
    statement: str
    source_ids: list[int] = Field(min_length=1)  # numbers of the retrieved passages used


class DraftStep(BaseModel):
    stage: Stage
    step: str
    source_ids: list[int] = Field(min_length=1)


class AnswerDraft(BaseModel):
    """The model cites passages by number. Code turns the numbers into real
    Citation objects, so the model can never invent a document or page."""
    summary: str
    findings: list[DraftFinding] = []
    roadmap: list[DraftStep] = []
    missing_information: list[MissingItem] = []
    confidence: Confidence = "low"
    confidence_reason: str = ""


ProblemType = Literal["residential_status", "nre_nro_accounts", "us_stocks_rsus",
                      "foreign_retirement", "foreign_asset_reporting", "other"]
ThreadStage = Literal["before_moving", "first_30_days", "first_year", "later", "unclear"]
Resolved = Literal["yes", "no", "unclear"]


class ThreadTag(BaseModel):
    problem_type: ProblemType
    stage: ThreadStage
    resolved: Resolved


class JudgeVerdict(BaseModel):
    label: Literal["supported", "partially_supported", "unsupported"]
    reason: str
