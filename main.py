import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.messages import AIMessage
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector

from database import get_vector_store
from exceptions import IngestError, RetrievalError, SettingsError
from ingest import ingest_text
from schemas import (
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_SUFFIXES = {".txt", ".md"}

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


def get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.google_api_key:
        raise RetrievalError("google_api_key is required")
    if not settings.llm_model:
        raise RetrievalError("llm_model is required")

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0,
        google_api_key=settings.google_api_key,
        max_retries=3,
        timeout=60,
    )


def _message_text(message: AIMessage | None) -> str:
    if message is None:
        raise RetrievalError("model returned no message")

    content = message.content
    if content is None:
        raise RetrievalError("model returned empty content")
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
    settings = get_settings()
    logger.info(
        "ask.start",
        extra={"question_len": len(request.question), "top_k": request.top_k},
    )

    try:
        vector_store: PGVector = get_vector_store(settings)
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_score(
                request.question,
                k=request.top_k,
            )
        )
    except SettingsError as err:
        logger.warning("ask.invalid_settings")
        raise HTTPException(
            status_code=500, detail="Server configuration error."
        ) from err
    except RetrievalError as err:
        logger.exception("ask.retrieval_failed")
        raise HTTPException(status_code=500, detail="Retrieval failed.") from err

    if not results:
        logger.info("ask.no_matches", extra={"question_len": len(request.question)})
        return QueryResponse(
            answer="I couldn't find any relevant documents in the database.",
            sources=[],
            model_used=settings.llm_model,
        )

    context_text = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    sources = [
        SourceDocument(
            content=doc.page_content,
            score=round(float(score), 4),
            metadata=doc.metadata,
        )
        for doc, score in results
    ]

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    chain = prompt | get_llm()

    try:
        response: AIMessage = await chain.ainvoke(
            {"context": context_text, "question": request.question}
        )
    except RetrievalError as err:
        logger.exception("ask.generation_failed")
        raise HTTPException(status_code=500, detail="Generation failed.") from err
    except (OSError, ValueError, RuntimeError) as err:
        logger.exception("ask.generation_failed")
        raise HTTPException(status_code=500, detail="Generation failed.") from err

    return QueryResponse(
        answer=_message_text(response),
        sources=sources,
        model_used=settings.llm_model,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_from_text(request: IngestTextRequest) -> IngestResponse:
    try:
        chunks_stored = ingest_text(request.text, source=request.source)
    except IngestError as err:
        logger.warning("ingest.text_failed", extra={"error": str(err)})
        raise HTTPException(status_code=400, detail=str(err)) from err
    except SettingsError as err:
        logger.warning("ingest.invalid_settings")
        raise HTTPException(
            status_code=500, detail="Server configuration error."
        ) from err

    return IngestResponse(chunks_stored=chunks_stored, source=request.source)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_from_file(file: UploadFile = File(...)) -> IngestResponse:
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    filename = file.filename or "upload.txt"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                f"Allowed: {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}"
            ),
        )

    try:
        raw = await file.read()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text.") from err

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks_stored = ingest_text(text, source=filename)
    except IngestError as err:
        logger.warning("ingest.file_failed", extra={"error": str(err)})
        raise HTTPException(status_code=400, detail=str(err)) from err
    except SettingsError as err:
        logger.warning("ingest.invalid_settings")
        raise HTTPException(
            status_code=500, detail="Server configuration error."
        ) from err

    return IngestResponse(chunks_stored=chunks_stored, source=filename)
