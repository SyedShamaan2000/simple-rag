import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.messages import AIMessage
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector

from database import get_vector_store
from ingest import ingest_text
from schemas import (
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

app = FastAPI(
    title="simple-rag",
    description="Backend API for the RAG knowledge assistant.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=3,
    timeout=60,
)

RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following pieces of retrieved context
to answer the user's question.

If the answer isn't in the context, say "I don't have enough information in
my database." Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
"""

MODEL_USED = "gemini-3.5-flash-lite"
ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md"}


def _message_text(message: AIMessage) -> str:
    """Gemini may return content as a string or a list of text blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="simple-rag")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest) -> QueryResponse:
    try:
        logger.info(f"Processing query: {request.question}")

        vector_store: PGVector = get_vector_store()
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_score(request.question, k=4)
        )

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant documents in the database.",
                sources=[],
                model_used=MODEL_USED,
            )

        context_text: str = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        sources: list[SourceDocument] = [
            SourceDocument(
                content=doc.page_content,
                score=round(float(score), 4),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]

        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
            RAG_PROMPT_TEMPLATE
        )
        chain = prompt | llm

        response: AIMessage = await chain.ainvoke(
            {"context": context_text, "question": request.question}
        )

        return QueryResponse(
            answer=_message_text(response),
            sources=sources,
            model_used=MODEL_USED,
        )

    except Exception as e:
        logger.error(f"Error during RAG process: {e!s}")
        raise HTTPException(status_code=500, detail="Internal system error.") from e


@app.post("/ingest", response_model=IngestResponse)
async def ingest_from_text(request: IngestTextRequest) -> IngestResponse:
    try:
        chunks_stored = ingest_text(request.text, source=request.source)
        return IngestResponse(chunks_stored=chunks_stored, source=request.source)
    except Exception as e:
        logger.error(f"Error during text ingest: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to ingest text.") from e


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_from_file(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "upload.txt"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}",
        )

    try:
        raw = await file.read()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 text.",
        ) from e

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks_stored = ingest_text(text, source=filename)
        return IngestResponse(chunks_stored=chunks_stored, source=filename)
    except Exception as e:
        logger.error(f"Error during file ingest: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to ingest file.") from e
