"""API tests. They run fully offline: conftest.py turns on MOCK_LLM."""
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "northwind_coffee_guidelines.md"


def _create_kit(client):
    with SAMPLE.open("rb") as f:
        res = client.post("/api/brand-kits", data={"name": "Northwind Coffee"},
                          files=[("files", (SAMPLE.name, f, "text/markdown"))])
    assert res.status_code == 201, res.text
    return res.json()


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["mock_llm"] is True


def test_create_and_list_brand_kit(client):
    kit = _create_kit(client)
    assert kit["chunks"] > 0 and kit["files"] == [SAMPLE.name]
    ids = [k["id"] for k in client.get("/api/brand-kits").json()]
    assert kit["id"] in ids
    assert client.get(f"/api/brand-kits/{kit['id']}").json()["name"] == "Northwind Coffee"


def test_rejects_unsupported_file(client):
    res = client.post("/api/brand-kits", data={"name": "Bad"},
                      files=[("files", ("logo.png", b"\x89PNG", "image/png"))])
    assert res.status_code == 400


def test_unknown_kit_returns_404(client):
    assert client.get("/api/brand-kits/000000000000").status_code == 404
    assert client.get("/api/brand-kits/../../etc").status_code == 404
    res = client.post("/api/generate", json={"brand_kit_id": "000000000000", "brief": "Launch our new cold brew."})
    assert res.status_code == 404


def test_generate_runs_review_and_revise_loop(client):
    kit = _create_kit(client)
    res = client.post("/api/generate", json={
        "brand_kit_id": kit["id"], "channel": "linkedin",
        "brief": "Announce our new oat-milk cold brew in Hyderabad cafes.",
    })
    assert res.status_code == 200, res.text
    body = res.json()
    # Mock reviewer fails the first draft and passes the revision.
    assert [s["step"] for s in body["history"]] == ["draft", "review", "revise", "review"]
    assert body["revisions"] == 1 and body["passed"] is True
    assert body["content"].startswith("REVISED DRAFT")
    assert body["sources"] and body["sources"][0]["file"] == SAMPLE.name


def test_generate_validates_input(client):
    kit = _create_kit(client)
    assert client.post("/api/generate", json={"brand_kit_id": kit["id"], "brief": "short"}).status_code == 422
    res = client.post("/api/generate", json={"brand_kit_id": kit["id"], "brief": "A long enough brief.", "channel": "tv"})
    assert res.status_code == 422
