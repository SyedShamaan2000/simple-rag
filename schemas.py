from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str
    sources: list[SourceDocument]
    model_used: str


class IngestTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    source: str = Field(default="uploaded", min_length=1, max_length=512)


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunks_stored: int = Field(ge=0)
    source: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: str
    service: str
