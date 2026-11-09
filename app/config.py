"""Settings read from environment variables (see .env.example)."""
import os
from pathlib import Path

# Load a local .env file if present (Azure uses App Settings instead).
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Mock mode runs the whole app without an API key (used by tests and CI).
MOCK_LLM = os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes") or not GOOGLE_API_KEY

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

# Agent behaviour
PASS_SCORE = int(os.getenv("PASS_SCORE", "8"))        # review score (0-10) needed to accept a draft
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))  # revision rounds after the first draft
TOP_K = int(os.getenv("TOP_K", "4"))                  # guideline chunks retrieved per request

# Upload limits
MAX_FILE_MB = 5
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
