import os

from dotenv import load_dotenv  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector


load_dotenv()

# Initialize Gemini Embeddings (Free Tier)
# This converts text into 768-dimensional vectors
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

CONNECTION_STRING: str | None = os.getenv("DATABASE_URL")
COLLECTION_NAME = "knowledge_base"


def get_vector_store() -> PGVector:
    """
    Returns the PGVector instance.
    It will automatically create the tables if they don't exist.
    """
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
