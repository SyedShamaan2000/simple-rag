import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_vector_store
from exceptions import IngestError

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def _split_and_store(docs: list[Document]) -> int:
    if not docs:
        raise IngestError("no documents to ingest")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Document] = text_splitter.split_documents(docs)
    if not chunks:
        logger.info("ingest.no_chunks", extra={"doc_count": len(docs)})
        return 0

    vector_store: PGVector = get_vector_store()
    try:
        vector_store.add_documents(chunks)
    except (OSError, ValueError) as err:
        logger.exception("ingest.store_failed", extra={"chunk_count": len(chunks)})
        raise IngestError("failed to store chunks") from err

    logger.info("ingest.chunk_saved", extra={"chunk_count": len(chunks)})
    return len(chunks)


def ingest_text(text: str, source: str = "uploaded") -> int:
    if text is None:
        raise IngestError("text is required")
    if not text.strip():
        raise IngestError("text must be non-empty")
    if source is None or not source.strip():
        raise IngestError("source must be non-empty")

    docs = [Document(page_content=text, metadata={"source": source.strip()})]
    return _split_and_store(docs)


def ingest_document(file_path: str) -> int:
    if file_path is None or not file_path.strip():
        raise IngestError("file_path is required")

    path = Path(file_path)
    if not path.is_file():
        raise IngestError(f"file not found: {path}")

    logger.info("ingest.start", extra={"path": str(path)})
    try:
        loader = TextLoader(str(path))
        docs: list[Document] = loader.load()
    except OSError as err:
        logger.exception("ingest.load_failed", extra={"path": str(path)})
        raise IngestError(f"failed to load file: {path}") from err

    if not docs:
        raise IngestError(f"file contained no documents: {path}")

    return _split_and_store(docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_document("knowledge.txt")
