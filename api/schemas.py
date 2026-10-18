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
    answer: str
    sources: list[SourceReference]
    citations_valid: bool
    has_citations: bool
    missing_citations: bool
    answer_quality: dict
    answer_quality={
        "is_acceptable": False,
        "has_answer": True,
        "has_citations": False,
        "has_sources": False,
        "is_abstention": True,
        "issues": ["no_sources"],
    },
