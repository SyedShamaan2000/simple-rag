import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_vector_store


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def ingest_document(file_path: str) -> None:
    logger.info(f"Starting ingestion for {file_path}")

    # 1. Load data
    loader = TextLoader(file_path)
    docs: list[Document] = loader.load()

    # 2. Split into chunks
    # We use 1000 chars with a 100 char overlap so context isn't lost at the edges
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks: list[Document] = text_splitter.split_documents(docs)

    # 3. Store in Postgres
    vector_store: PGVector = get_vector_store()
    vector_store.add_documents(chunks)

    logger.info(f"Successfully stored {len(chunks)} chunks in Postgres.")


if __name__ == "__main__":
    with Path("knowledge.txt").open():
        ingest_document("knowledge.txt")
