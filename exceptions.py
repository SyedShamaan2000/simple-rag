class SimpleRagError(Exception):
    """Base error for application-level failures."""


class SettingsError(SimpleRagError):
    """Raised when required configuration is missing or invalid."""


class IngestError(SimpleRagError):
    """Raised when document ingestion fails."""


class RetrievalError(SimpleRagError):
    """Raised when retrieval or generation fails."""
