from typing import Any

from pydantic import BaseModel


# This is what the user sends TO the API
class QueryRequest(BaseModel):
    question: str


# This represents a single chunk found in Postgres
class SourceDocument(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


# This is what the API sends BACK to the user
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    model_used: str
