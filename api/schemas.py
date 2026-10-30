from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask the knowledge base",
    )

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value):
        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("Question cannot be empty or whitespace.")

        return value


class SourceReference(BaseModel):
    citation: str = Field(pattern=r"^S\d+$")
    source: str = Field(min_length=1)
    page: int = Field(ge=1)


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: list[SourceReference]
    citations_valid: bool
    has_citations: bool
    missing_citations: bool
    answer_quality: dict


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)
    rating: str
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("query_id", "question", "answer", mode="before")
    @classmethod
    def validate_required_text(cls, value):
        if isinstance(value, str):
            value = value.strip()

            if not value:
                raise ValueError("This field cannot be empty or whitespace.")

        return value

    @field_validator("rating", mode="before")
    @classmethod
    def validate_rating(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()

        if value not in {"up", "down"}:
            raise ValueError("Rating must be either 'up' or 'down'.")

        return value

    @field_validator("comment", mode="before")
    @classmethod
    def normalize_comment(cls, value):
        if isinstance(value, str):
            value = value.strip() or None

        return value


class FeedbackResponse(BaseModel):
    feedback_id: int
    query_id: str
    rating: str
    message: str
