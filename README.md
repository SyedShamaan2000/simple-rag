Project Structure

ml-rag-project/
├── .env                # API Keys (Git ignored)
├── .env.reference      # Template for .env
├── docker-compose.yml  # Database config
├── main.py             # FastAPI Entry point
├── schemas.py          # Pydantic models
├── database.py         # DB connection & Vector Store logic
└── ingest.py           # Script to load your documents