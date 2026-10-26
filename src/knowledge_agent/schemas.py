from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


class Citation(BaseModel):
    source: str
    page: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    retrieved_chunks: int = 0


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str
