import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas import QueryRequest, QueryResponse, SourceDocument

# Internal imports from our previous steps
from database import get_vector_store


# 1. Setup Logging & Environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pro-Level Free RAG API")

# 2. Initialize the LLM (Gemini Flash)
# We set temperature=0 for "Fact-based" answers (less creative, more accurate)
# In main.py
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Switch from 2.0-flash to 2.5-flash-lite
    temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 3. Define the Prompt Template
# This is where we "ground" the AI to prevent hallucinations.
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following pieces of retrieved context 
to answer the user's question. 

If the answer isn't in the context, say "I don't have enough information in my database." 
Do not make up facts.

Context:
{context}

Question: 
{question}

Answer:
"""


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest) -> QueryResponse:
    try:
        logger.info(f"Processing query: {request.question}")

        # A. RETRIEVAL: Search Postgres for top 4 matches
        vector_store = get_vector_store()
        # similarity_search_with_score returns a tuple (Document, float_score)
        results = vector_store.similarity_search_with_score(request.question, k=4)

        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant documents in the database.",
                sources=[],
                model_used="gemini-2.0-flash",
            )

        # B. FORMATTING: Prepare context string and source list
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        sources = [
            SourceDocument(
                content=doc.page_content,
                score=round(float(score), 4),
                metadata=doc.metadata,
            )
            for doc, score in results
        ]

        # C. GENERATION: Send to Gemini
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm

        response = await chain.ainvoke(
            {"context": context_text, "question": request.question}
        )

        return QueryResponse(
            answer=response.content, sources=sources, model_used="gemini-2.0-flash"
        )

    except Exception as e:
        logger.error(f"Error during RAG process: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal system error.")
