from __future__ import annotations

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
            "source code",
            "proprietary",
            "public domain",
            "technical and non-technical",
        ],
        obligations_receiving_party=[
            "strict confidence",
            "restrict access",
            "not disclose",
            "not publish",
            "not copy",
            "sole and exclusive benefit",
        ],
        governing_law=[
            "governed by",
            "state of georgia",
            "governing law",
            "conflict of law",
        ],
    )


def score_nda_summary(response_text: str, rubric: AccuracyRubric | None = None) -> Dict[str, float]:
    """
    Score whether a generated NDA summary covers the 3 required concepts.

    The score is transparent and demo-friendly for classroom settings:
      - Each concept gets 1.0 if any associated keyword is present, else 0.0
      - Overall accuracy is the average of the 3 concept scores
    """
    rubric = rubric or default_rubric()
    response_lower = response_text.lower()

    concept_keywords = {
        "confidential_information": rubric.confidential_information,
        "obligations_receiving_party": rubric.obligations_receiving_party,
        "governing_law": rubric.governing_law,
    }

    per_concept = {}
    for concept, keywords in concept_keywords.items():
        matched = any(keyword in response_lower for keyword in keywords)
        per_concept[concept] = 1.0 if matched else 0.0

    overall = sum(per_concept.values()) / len(per_concept)

    return {
        "accuracy": overall,
        **{f"{k}_score": v for k, v in per_concept.items()},
    }
