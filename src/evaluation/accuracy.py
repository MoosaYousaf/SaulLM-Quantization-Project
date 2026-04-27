from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class AccuracyRubric:
    """Gold-standard statements used to semantically score core NDA concepts."""

    confidential_information: str
    obligations_receiving_party: str
    governing_law: str


def default_rubric() -> AccuracyRubric:
    return AccuracyRubric(
        confidential_information=(
            "Confidential Information includes non-public technical and non-technical proprietary material, "
            "such as source code and business information, but excludes anything that is publicly available."
        ),
        obligations_receiving_party=(
            "The receiving party must keep disclosed information in strict confidence, limit access, and avoid "
            "disclosing, publishing, or copying it except for the agreement's permitted purpose."
        ),
        governing_law=(
            "The NDA is governed by the laws of the State of Georgia, without applying conflict-of-law principles."
        ),
    )


_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model


def _to_unit_interval(value: float) -> float:
    return float(max(0.0, min(1.0, (value + 1.0) / 2.0)))


def _concept_similarity_scores(response_text: str, rubric: AccuracyRubric) -> Tuple[float, float, float]:
    model = _get_model()

    gold_sentences = [
        rubric.confidential_information,
        rubric.obligations_receiving_party,
        rubric.governing_law,
    ]

    response_embedding = model.encode([response_text], normalize_embeddings=True)
    gold_embeddings = model.encode(gold_sentences, normalize_embeddings=True)

    similarities = cosine_similarity(response_embedding, gold_embeddings)[0]
    return tuple(_to_unit_interval(score) for score in similarities)


def score_nda_summary(response_text: str, rubric: AccuracyRubric | None = None) -> Dict[str, float]:
    """Score NDA summary quality by semantic similarity to concept gold standards."""
    rubric = rubric or default_rubric()

    confidential_score, obligations_score, governing_law_score = _concept_similarity_scores(response_text, rubric)

    overall = float((confidential_score + obligations_score + governing_law_score) / 3.0)

    # Keep legacy keys for CSV compatibility in run_benchmark.py.
    return {
        "accuracy": overall,
        "confidential_information_score": confidential_score,
        "obligations_receiving_party_score": obligations_score,
        "governing_law_score": governing_law_score,
        "confidential_information_matched_keywords": confidential_score * 100.0,
        "obligations_receiving_party_matched_keywords": obligations_score * 100.0,
        "governing_law_matched_keywords": governing_law_score * 100.0,
        "confidential_information_total_keywords": 100.0,
        "obligations_receiving_party_total_keywords": 100.0,
        "governing_law_total_keywords": 100.0,
    }
