"""Tool: calculate weighted totals and rank suppliers.

Deterministic scoring so every ranking is reproducible and auditable:
the LLM judges per-criterion scores from evidence, the arithmetic and
ranking rules live here in code.
"""

import math
from functools import cmp_to_key

from pydantic import BaseModel, Field, field_validator, model_validator

TECHNICAL_CRITERION = "Technical Solution"
FINANCIAL_CRITERION = "Financial"
TIE_BAND = 1.0  # totals within one point are ties per the evaluation methodology


class Scorecard(BaseModel):
    supplier: str
    scores: dict[str, float] = Field(
        description="criterion name -> 0-100 score", min_length=1
    )

    @field_validator("scores")
    @classmethod
    def scores_in_range(cls, scores):
        bad = {k: v for k, v in scores.items() if not 0 <= v <= 100}
        if bad:
            raise ValueError(f"scores must be between 0 and 100, got {bad}")
        return scores


class WeightedScoreInput(BaseModel):
    """Validated input for calculate_weighted_score."""

    weights: dict[str, float] = Field(
        description="criterion name -> weight in percent; must sum to 100",
        min_length=1,
    )
    scorecards: list[Scorecard] = Field(min_length=1)
    disqualified: list[str] = Field(
        default_factory=list,
        description="Suppliers failing a mandatory requirement (ranked last, no award)",
    )

    @field_validator("weights")
    @classmethod
    def weights_sum_to_100(cls, weights):
        total = sum(weights.values())
        if not math.isclose(total, 100.0, abs_tol=0.01):
            raise ValueError(f"criterion weights must sum to 100, got {total}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("criterion weights cannot be negative")
        return weights

    @model_validator(mode="after")
    def scorecards_cover_all_criteria(self):
        criteria = set(self.weights)
        for card in self.scorecards:
            missing = criteria - set(card.scores)
            if missing:
                raise ValueError(
                    f"scorecard for '{card.supplier}' is missing criteria {sorted(missing)}"
                )
        return self


def calculate_weighted_score(payload: WeightedScoreInput) -> dict:
    """Weighted totals, per-criterion breakdown and final ranking."""
    results = []
    for card in payload.scorecards:
        breakdown = {}
        total = 0.0
        for criterion, weight in payload.weights.items():
            score = card.scores[criterion]
            weighted = score * weight / 100.0
            total += weighted
            breakdown[criterion] = {
                "score": round(score, 2),
                "weight": weight,
                "weighted": round(weighted, 2),
            }
        results.append(
            {
                "supplier": card.supplier,
                "weighted_total": round(total, 2),
                "breakdown": breakdown,
                "disqualified": card.supplier in payload.disqualified,
            }
        )

    # Qualified bids rank first. Totals within TIE_BAND count as tied per the
    # evaluation methodology: the higher Technical Solution score wins, then
    # the lower TCO — which, under the financial formula (100 x lowest/TCO),
    # is the higher Financial score. (Pairwise rule, applied as a comparator.)
    def _criterion(r, name):
        return r["breakdown"].get(name, {}).get("score", 0.0)

    def compare(a, b):
        if a["disqualified"] != b["disqualified"]:
            return 1 if a["disqualified"] else -1
        gap = a["weighted_total"] - b["weighted_total"]
        if abs(gap) > TIE_BAND:
            return -1 if gap > 0 else 1
        for name in (TECHNICAL_CRITERION, FINANCIAL_CRITERION):
            delta = _criterion(a, name) - _criterion(b, name)
            if delta:
                return -1 if delta > 0 else 1
        return -1 if gap > 0 else (1 if gap < 0 else 0)

    ranked = sorted(results, key=cmp_to_key(compare))
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank

    return {
        "ranking": ranked,
        "best_qualified": next(
            (r["supplier"] for r in ranked if not r["disqualified"]), None
        ),
    }
