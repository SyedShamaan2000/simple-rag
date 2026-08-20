from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class SourceDocument(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    model_used: str


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "uploaded"


class IngestResponse(BaseModel):
    chunks_stored: int
    source: str


class HealthResponse(BaseModel):
    status: str
    service: str
