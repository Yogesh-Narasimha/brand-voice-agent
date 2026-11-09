import os
import tempfile

# Must be set before the app is imported.
os.environ["MOCK_LLM"] = "1"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="bva-test-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)
