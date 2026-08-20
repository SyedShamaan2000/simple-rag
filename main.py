import logging
import os

from fastapi import FastAPI, HTTPException
from langchain.messages import AIMessage
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector

# Internal imports from our previous steps
from database import get_vector_store
from schemas import QueryRequest, QueryResponse, SourceDocument

# 1. Setup Logging & Environment
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

app = FastAPI(title="Pro-Level Free RAG API")

# 2. Initialize the LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=3,  # Automatically retry up to 3 times on 429 errors
    timeout=60,  # Wait up to 60 seconds for a response
)

# 3. Define the Prompt Template
# This is where we "ground" the AI to prevent hallucinations.
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


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest) -> QueryResponse:
    try:
        logger.info(f"Processing query: {request.question}")

        # A. RETRIEVAL: Search Postgres for top 4 matches
        vector_store: PGVector = get_vector_store()
        # similarity_search_with_score returns a tuple (Document, float_score)
        results: list[tuple[Document, float]] = (
            vector_store.similarity_search_with_score(request.question, k=4)
        )

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant documents in the database.",
                sources=[],
                model_used=MODEL_USED,
            )

        # B. FORMATTING: Prepare context string and source list
        context_text: str = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        sources: list[SourceDocument] = [
            SourceDocument(
                content=doc.page_content,
                score=round(float(score), 4),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]

        # C. GENERATION: Send to Gemini
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
