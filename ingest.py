import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_vector_store

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def _split_and_store(docs: list[Document]) -> int:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Document] = text_splitter.split_documents(docs)
    if not chunks:
        return 0

    vector_store: PGVector = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def ingest_text(text: str, source: str = "uploaded") -> int:
    docs = [Document(page_content=text, metadata={"source": source})]
    stored = _split_and_store(docs)
    logger.info(f"Stored {stored} chunks from source={source}")
    return stored


def ingest_document(file_path: str) -> int:
    logger.info(f"Starting ingestion for {file_path}")
    loader = TextLoader(file_path)
    docs: list[Document] = loader.load()
    stored = _split_and_store(docs)
    logger.info(f"Successfully stored {stored} chunks in Postgres.")
    return stored


if __name__ == "__main__":
    with Path("knowledge.txt").open():
        ingest_document("knowledge.txt")
