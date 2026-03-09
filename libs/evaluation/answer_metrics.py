"""Answer quality metrics: faithfulness, citation accuracy, unsupported claims."""

from __future__ import annotations

from dataclasses import dataclass, field

from libs.contracts.generation import GeneratedAnswer


@dataclass(frozen=True)
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    score: float  # [0.0, 1.0]
    total_claims: int
    supported_claims: int
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CitationAccuracyResult:
    """Result of citation accuracy evaluation."""
    score: float  # [0.0, 1.0]
    total_citations: int
    valid_citations: int
    invalid_citations: list[str] = field(default_factory=list)


def evaluate_citation_accuracy(
    answer: GeneratedAnswer,
    context_chunk_ids: set[str],
) -> CitationAccuracyResult:
    """Check whether citations reference chunks that were in the context.

    A citation is valid if its chunk_id exists in the provided context.

    Score = valid_count / total_count (0.0 if no citations).
    """
    if not answer.citations:
        return CitationAccuracyResult(
            score=1.0,  # No citations = no invalid citations
            total_citations=0,
            valid_citations=0,
        )

    valid = 0
    invalid: list[str] = []
    for citation in answer.citations:
        if citation.chunk_id in context_chunk_ids:
            valid += 1
        else:
            invalid.append(citation.chunk_id)

    total = len(answer.citations)
    return CitationAccuracyResult(
        score=valid / total,
        total_citations=total,
        valid_citations=valid,
        invalid_citations=invalid,
    )


def evaluate_faithfulness(
    claims: list[str],
    supported_chunk_ids: set[str],
    claim_to_chunk: dict[str, str],
) -> FaithfulnessResult:
    """Evaluate whether claims are supported by context chunks.

    Takes a mapping of claim text -> chunk_id that supposedly supports it.
    A claim is supported if its mapped chunk_id is in the supported set.

    Score = supported_count / total_count (0.0 if no claims).

    This is a structural check. For semantic faithfulness evaluation,
    integrate an LLM-as-judge later.
    """
    if not claims:
        return FaithfulnessResult(
            score=1.0,
            total_claims=0,
            supported_claims=0,
        )

    supported = 0
    unsupported: list[str] = []
    for claim in claims:
        chunk_id = claim_to_chunk.get(claim)
        if chunk_id and chunk_id in supported_chunk_ids:
            supported += 1
        else:
            unsupported.append(claim)

    return FaithfulnessResult(
        score=supported / len(claims),
        total_claims=len(claims),
        supported_claims=supported,
        unsupported_claims=unsupported,
    )


def detect_unsupported_claims(
    answer: GeneratedAnswer,
    context_chunk_ids: set[str],
) -> list[str]:
    """Return claims whose citations reference chunks not in context.

    A claim is unsupported if its citation's chunk_id is not in context_chunk_ids.
    """
    unsupported: list[str] = []
    for citation in answer.citations:
        if citation.chunk_id not in context_chunk_ids:
            unsupported.append(citation.claim)
    return unsupported
