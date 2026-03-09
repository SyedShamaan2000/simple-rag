from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# This is what the user sends TO the API
class QueryRequest(BaseModel):
    question: str

# This represents a single chunk found in Postgres
class SourceDocument(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

# This is what the API sends BACK to the user
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    model_used: str