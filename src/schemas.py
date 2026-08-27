"""Typed contracts shared by the supervisor, specialists, tools and UI.

Every hand-off between agents is one of these Pydantic models, so the
information flowing through the graph is always structured and validated.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence sourcing
# ---------------------------------------------------------------------------

class SourceRef(BaseModel):
    """Where a piece of retrieved evidence came from."""

    file: str = Field(description="Source file name, e.g. 'bid_alphatech.md'")
    section: str = Field(default="", description="Section heading inside the file")
    chunk_id: str = Field(default="", description="Vector-store chunk identifier")


# ---------------------------------------------------------------------------
# Requirement Specialist output
# ---------------------------------------------------------------------------

class Criterion(BaseModel):
    """One weighted evaluation criterion from the tender."""

    name: str = Field(description="Criterion name, e.g. 'Technical solution'")
    weight: float = Field(description="Weight as a percentage, e.g. 40 for 40%")
    description: str = Field(default="", description="What the criterion measures")


class Requirement(BaseModel):
    """One tender requirement a bid must address."""

    requirement_id: str = Field(description="Stable id, e.g. 'M1' or 'S3'")
    title: str = Field(description="Short requirement title")
    description: str = Field(description="What the supplier must provide or prove")
    kind: Literal["mandatory", "scored"] = Field(
        description="'mandatory' = pass/fail gate, 'scored' = contributes to score"
    )
    criterion: str = Field(
        default="",
        description="For scored requirements: the evaluation criterion it belongs to",
    )
    source: Optional[SourceRef] = Field(
        default=None, description="Where in the tender this requirement was found"
    )


class RequirementSet(BaseModel):
    """Structured result of tender analysis."""

    tender_ref: str = Field(description="Tender reference, e.g. 'RFP-2026-014'")
    tender_title: str
    summary: str = Field(description="2-3 sentence summary of what is being procured")
    requirements: list[Requirement]
    criteria: list[Criterion] = Field(description="Weighted evaluation criteria")


# ---------------------------------------------------------------------------
# Supplier Evidence Specialist output
# ---------------------------------------------------------------------------

ComplianceStatus = Literal["compliant", "partial", "missing"]


class EvidenceItem(BaseModel):
    """A single requirement checked against a single supplier bid."""

    requirement_id: str
    supplier: str
    status: ComplianceStatus
    evidence_quote: str = Field(
        default="",
        description="Short quote from the bid that supports the status ('' if missing)",
    )
    source: Optional[SourceRef] = None
    note: str = Field(default="", description="Assessor note explaining the judgement")


class SupplierFindings(BaseModel):
    """Full compliance assessment for one supplier."""

    supplier: str
    items: list[EvidenceItem]
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    """Structured result of the evidence audit across all suppliers."""

    findings: list[SupplierFindings]


# ---------------------------------------------------------------------------
# Comparison Specialist output
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=0, le=100, description="0-100 score for this criterion")
    justification: str = Field(description="Evidence-based reason for the score")


class SupplierScorecard(BaseModel):
    supplier: str
    scores: list[CriterionScore]


class RankedSupplier(BaseModel):
    supplier: str
    weighted_total: float
    rank: int
    disqualified: bool = Field(
        default=False, description="True when a mandatory requirement is missing"
    )


class ComparisonResult(BaseModel):
    """Structured result of the supplier comparison."""

    scorecards: list[SupplierScorecard]
    ranking: list[RankedSupplier]
    recommendation: str = Field(description="Which supplier to select and why")
    caveats: list[str] = Field(
        default_factory=list,
        description="Risks, conditions or missing evidence the buyer should resolve",
    )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

AgentName = Literal[
    "requirement_specialist",
    "evidence_specialist",
    "comparison_specialist",
    "finalize",
]


class SupervisorPlan(BaseModel):
    """Initial plan the supervisor produces before delegating."""

    objective: str = Field(description="One-sentence statement of the case objective")
    steps: list[str] = Field(description="Ordered plan steps in plain language")


class RouteDecision(BaseModel):
    """One routing decision in the plan-and-execute loop."""

    next_agent: AgentName
    reasoning: str = Field(description="Why this agent should act next")


class FinalReport(BaseModel):
    """The combined business result returned to the user."""

    tender_ref: str
    executive_summary: str
    recommended_supplier: str
    key_risks: list[str]
    missing_evidence_summary: list[str] = Field(
        description="Outstanding items the procurement team must chase per supplier"
    )
    next_steps: list[str]
