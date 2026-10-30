from src.feedback.database import insert_feedback


def submit_feedback(
    query_id: str,
    question: str,
    answer: str,
    rating: str,
    comment: str | None = None,
) -> dict:
    """Validate and save user feedback."""

    query_id = query_id.strip()
    question = question.strip()
    answer = answer.strip()
    rating = rating.strip().lower()

    if not query_id:
        raise ValueError("Query ID cannot be empty.")

    if not question:
        raise ValueError("Question cannot be empty.")

    if not answer:
        raise ValueError("Answer cannot be empty.")

    if rating not in {"up", "down"}:
        raise ValueError("Rating must be either 'up' or 'down'.")

    if comment is not None:
        comment = comment.strip() or None

        if comment and len(comment) > 2000:
            raise ValueError("Comment cannot exceed 2000 characters.")

    feedback_id = insert_feedback(
        query_id=query_id,
        question=question,
        answer=answer,
        rating=rating,
        comment=comment,
    )

    return {
        "feedback_id": feedback_id,
        "query_id": query_id,
        "rating": rating,
        "message": "Feedback submitted successfully.",
    }
