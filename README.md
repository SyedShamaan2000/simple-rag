Project Structure

ml-rag-project/
├── .env                # API Keys (Git ignored)
├── .env.reference      # Template for .env
├── docker-compose.yml  # Database config
├── main.py             # FastAPI Entry point
├── schemas.py          # Pydantic models
├── database.py         # DB connection & Vector Store logic
└── ingest.py           # Script to load your documents

To run the project:
1. Set up your .env file with the required API keys and database URL.
2. Start the database using Docker: `docker-compose up -d`
3. Ingest your documents: `uv run ingest.py`
4. Start the FastAPI server: `uv run uvicorn main:app --reload --port 8000 --host 0.0.0.0`