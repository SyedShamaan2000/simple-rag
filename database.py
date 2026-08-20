import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from exceptions import RetrievalError
from settings import Settings, get_settings

logger = logging.getLogger(__name__)


def get_embeddings(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    if settings is None:
        settings = get_settings()
    if not settings.google_api_key:
        raise RetrievalError("google_api_key is required")
    if not settings.embedding_model:
        raise RetrievalError("embedding_model is required")

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def get_vector_store(settings: Settings | None = None) -> PGVector:
    if settings is None:
        settings = get_settings()
    if not settings.database_url:
        raise RetrievalError("database_url is required")
    if not settings.collection_name:
        raise RetrievalError("collection_name is required")

    logger.info(
        "vector_store.init",
        extra={"collection": settings.collection_name},
    )
    return PGVector(
        embeddings=get_embeddings(settings),
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )
