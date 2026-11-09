"""Brand kits: guideline documents split into chunks and stored in a FAISS index on disk."""
import io
import json
import re
import uuid
from datetime import datetime, timezone

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config
from app.llm import get_embeddings

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
_cache: dict[str, FAISS] = {}


def _kit_dir(kit_id: str):
    if not re.fullmatch(r"[a-f0-9]{12}", kit_id):
        raise KeyError(kit_id)
    return config.DATA_DIR / "brand_kits" / kit_id


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def create_kit(name: str, files: list[tuple[str, bytes]]) -> dict:
    docs = []
    for filename, data in files:
        text = extract_text(filename, data).strip()
        for i, chunk in enumerate(_splitter.split_text(text)):
            docs.append(Document(page_content=chunk, metadata={"source": filename, "chunk": i}))
    if not docs:
        raise ValueError("The uploaded files contain no readable text.")

    kit_id = uuid.uuid4().hex[:12]
    store = FAISS.from_documents(docs, get_embeddings())
    path = _kit_dir(kit_id)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    meta = {
        "id": kit_id,
        "name": name,
        "files": [f for f, _ in files],
        "chunks": len(docs),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / "meta.json").write_text(json.dumps(meta))
    _cache[kit_id] = store
    return meta


def get_kit_meta(kit_id: str) -> dict:
    path = _kit_dir(kit_id) / "meta.json"
    if not path.exists():
        raise KeyError(kit_id)
    return json.loads(path.read_text())


def list_kits() -> list[dict]:
    root = config.DATA_DIR / "brand_kits"
    if not root.exists():
        return []
    kits = [json.loads(p.read_text()) for p in root.glob("*/meta.json")]
    return sorted(kits, key=lambda k: k["created_at"], reverse=True)


def retrieve(kit_id: str, query: str, k: int = config.TOP_K) -> list[Document]:
    if kit_id not in _cache:
        path = _kit_dir(kit_id)
        if not (path / "index.faiss").exists():
            raise KeyError(kit_id)
        # Safe here: we only load indexes this app wrote itself.
        _cache[kit_id] = FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)
    return _cache[kit_id].similarity_search(query, k=k)
