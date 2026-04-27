from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AccuracyRubric:
    """Keyword-based rubric to score whether core NDA summary points are present."""

    confidential_information: List[str]
    obligations_receiving_party: List[str]
    governing_law: List[str]


def default_rubric() -> AccuracyRubric:
    return AccuracyRubric(
        confidential_information=[
            "confidential information",
            "confidential",
            "source code",
            "proprietary",
            "public domain",
            "technical and non-technical",
            "technical",
        ],
        obligations_receiving_party=[
            "receiving party",
            "strict confidence",
            "strictest confidence",
            "restrict access",
            "not disclose",
            "not publish",
            "not copy",
            "sole and exclusive benefit",
            "obligations",
        ],
        governing_law=[
            "governed by",
            "governing",
            "state of georgia",
            "governing law",
            "conflict of law",
            "law",
        ],
    )


def _contains_keyword(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword.lower())
    if " " in keyword:
        return keyword.lower() in text
    return re.search(rf"\b{escaped}\w*\b", text) is not None


def _concept_coverage(text: str, keywords: List[str]) -> Dict[str, float]:
    matched_keywords = [kw for kw in keywords if _contains_keyword(text, kw)]
    unique_matches = len(set(matched_keywords))
    total_keywords = len(keywords)
    target_matches_for_full_score = 3
    coverage = min(1.0, unique_matches / target_matches_for_full_score) if total_keywords else 0.0
    return {
        "coverage": coverage,
        "matched": float(unique_matches),
        "total": float(total_keywords),
    }


def score_nda_summary(response_text: str, rubric: AccuracyRubric | None = None) -> Dict[str, float]:
    """
    Score whether a generated NDA summary covers the 3 required concepts.

    The score is transparent and demo-friendly for classroom settings:
      - Each concept gets a coverage score in [0,1] based on matched rubric keywords
      - Coverage saturates at 1.0 once 3+ concept keywords are matched
      - Overall accuracy is the mean of the 3 concept coverage scores
    """
    rubric = rubric or default_rubric()
    response_lower = response_text.lower()

    concept_keywords = {
        "confidential_information": rubric.confidential_information,
        "obligations_receiving_party": rubric.obligations_receiving_party,
        "governing_law": rubric.governing_law,
    }

    per_concept = {}
    per_concept_matched = {}
    per_concept_total = {}
    for concept, keywords in concept_keywords.items():
        coverage = _concept_coverage(response_lower, keywords)
        per_concept[concept] = coverage["coverage"]
        per_concept_matched[f"{concept}_matched_keywords"] = coverage["matched"]
        per_concept_total[f"{concept}_total_keywords"] = coverage["total"]

    overall = sum(per_concept.values()) / len(per_concept)

    return {
        "accuracy": overall,
        **{f"{k}_score": v for k, v in per_concept.items()},
        **per_concept_matched,
        **per_concept_total,
    }
