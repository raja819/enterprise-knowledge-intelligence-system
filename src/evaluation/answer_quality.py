import re

ABSTENTION_PHRASES = (
    "do not contain enough information",
    "does not contain enough information",
    "could not find supporting documents",
    "i cannot determine",
    "i can't determine",
    "i don't have enough information",
    "not mentioned in the provided document excerpts",
    "no answer can be given based on the available information",
    "do not contain information",
    "does not contain information",
    "do not contain any information",
    "does not contain any information",
    "not contained in the provided documents",
    "not available in the provided documents",
    "not available in the provided excerpts",
    "not mentioned in the provided excerpts",
    "not specified in the provided documents",
    "cannot be determined from the provided information",
    "cannot be determined from the available information",
    "insufficient information",
)


def evaluate_answer_quality(
    answer: str,
    sources: list[dict],
) -> dict:
    """
    Perform basic, deterministic quality checks.

    Note:
    This checks answer structure and citation consistency.
    It does not prove factual correctness or semantic grounding.
    """
    if not isinstance(answer, str):
        raise ValueError("Answer must be a string.")

    if not isinstance(sources, list):
        raise ValueError("Sources must be a list.")

    answer = answer.strip()

    if not answer:
        return {
            "is_acceptable": False,
            "has_answer": False,
            "has_citations": False,
            "has_sources": bool(sources),
            "is_abstention": False,
            "issues": ["empty_answer"],
        }

    # Detect citations such as [S1], [S2], etc.
    cited_labels = set(
        re.findall(r"\[(S\d+)\]", answer)
    )

    available_labels = {
        source["citation"]
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("citation"), str)
        and re.fullmatch(r"S\d+", source["citation"])
    }

    issues = []

    if not sources:
        issues.append("no_sources")

    if not cited_labels:
        issues.append("missing_citations")

    unknown_labels = sorted(cited_labels - available_labels)

    if unknown_labels:
        issues.append("unknown_citations")

    # Remove citations before checking abstention language.
    answer_without_citations = re.sub(
        r"\[\s*S\d+\s*\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    answer_without_citations = re.sub(
        r"\s+",
        " ",
        answer_without_citations,
    ).strip().lower()

    # Split the answer into sentences.
    answer_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            answer_without_citations,
        )
        if sentence.strip()
    ]

    # Find sentences that indicate insufficient evidence.
    abstention_sentences = [
        sentence
        for sentence in answer_sentences
        if any(
            phrase in sentence
            for phrase in ABSTENTION_PHRASES
        )
    ]

    # The answer is a plain abstention only if every sentence
    # expresses that the available information is insufficient.
    is_abstention = (
        bool(abstention_sentences)
        and len(abstention_sentences) == len(answer_sentences)
    )

    return {
        "is_acceptable": (
            bool(answer)
            and bool(sources)
            and (
                (
                    is_abstention
                    and not cited_labels
                )
                or (
                    bool(cited_labels)
                    and not unknown_labels
                )
            )
        ),
        "has_answer": bool(answer),
        "has_citations": bool(cited_labels),
        "has_sources": bool(sources),
        "is_abstention": is_abstention,
        "unknown_citations": unknown_labels,
        "issues": issues,
    }