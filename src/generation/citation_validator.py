import re


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def validate_citations(
    answer: str,
    sources: list[dict],
) -> dict:
    """
    Check citation labels against supplied source references.

    This validates citation-label consistency, not whether a
    source semantically supports every factual claim.
    """

    if not isinstance(answer, str):
        raise ValueError("Answer must be a string.")

    if not isinstance(sources, list):
        raise ValueError("Sources must be a list.")

    valid_labels = {
        source.get("citation")
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("citation"), str)
        and re.fullmatch(r"S\d+", source.get("citation"))
    }

    cited_labels = set(CITATION_PATTERN.findall(answer))
    unknown_citations = sorted(cited_labels - valid_labels)
    has_citations = bool(cited_labels)

    # When sources are supplied, require at least one citation.
    missing_citations = bool(sources) and not has_citations

    return {
        "is_valid": (
            not unknown_citations
            and not missing_citations
        ),
        "cited_labels": sorted(cited_labels),
        "unknown_citations": unknown_citations,
        "has_citations": has_citations,
        "missing_citations": missing_citations,
    }