# Brand Voice Content Agent

A GenAI web app that writes **on-brand marketing content**. Upload a brand's guideline documents, give a short brief,
and a LangGraph agent drafts the content, reviews it against the retrieved guidelines, and revises it until it passes.

**Live demo:** `https://<your-app-name>.azurewebsites.net` · **API docs:** `/docs`

## How it works

```
 Upload guidelines (.md/.txt/.pdf) ──► chunk ──► Gemini embeddings ──► FAISS index (per brand kit)

 Brief + channel ──► LangGraph agent
                      retrieve ─► draft ─► review ──(score ≥ 8 or max revisions)──► final content
                                             ▲    │
                                             └ revise
```

- **Retrieval (RAG):** the brief is used to fetch the most relevant guideline chunks, so every draft and review is
  grounded in the brand's own rules, and the response lists which chunks were used.
- **Agent loop (LangGraph):** a reviewer step returns a JSON score (0–10) with concrete issues; below the pass score,
  a revise step fixes them. The loop is capped (default 2 revisions) to bound cost and latency.
- **Channels:** LinkedIn, email, Instagram, search ad and product description, each with its own length and format rules.
- **REST API (FastAPI):** validated requests (Pydantic), clear 400/404/413/422/502 errors, and request logging.
- **Mock mode:** with no API key (or `MOCK_LLM=1`) the app uses a fake model and embeddings, so tests and CI run offline.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status and model in use |
| POST | `/api/brand-kits` | Multipart upload: `name` + one or more `files` → creates a brand kit |
| GET | `/api/brand-kits` | List brand kits |
| GET | `/api/brand-kits/{id}` | Brand kit details |
| POST | `/api/generate` | JSON `{brand_kit_id, brief, channel}` → content, review score, revisions, sources, agent steps |

## Run locally (Windows PowerShell)

```powershell
cd brand-voice-agent
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt

copy .env.example .env      # then open .env and paste your key
pytest -q                   # 6 tests, run offline
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, create a brand kit from `samples/northwind_coffee_guidelines.md`, and generate a LinkedIn post.

Get a free Gemini API key at https://aistudio.google.com/apikey. If you see a "model not found" error, set `CHAT_MODEL`
in `.env` to a current Flash model name from https://ai.google.dev/gemini-api/docs/models.

## Deploy to Azure App Service (free tier) with GitHub Actions

1. **Azure account:** sign up for *Azure for Students* (https://azure.microsoft.com/free/students) with your college email (no credit card needed).
2. **Create the Web App:** Azure Portal → *Create a resource* → *Web App*.
   Publish: *Code* · Runtime: *Python 3.12* · OS: *Linux* · Region: *Central India* · Pricing plan: *Free F1*.
3. **Configure it** (Web App → *Settings*):
   - *Environment variables* → add `GOOGLE_API_KEY` = your key, `SCM_DO_BUILD_DURING_DEPLOYMENT` = `true`, `DATA_DIR` = `/home/data`.
   - *Configuration* → *General settings* → Startup command: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
     On the same page turn **SCM Basic Auth Publishing Credentials** *On* (needed for the publish profile). Save.
4. **Connect GitHub:** Web App → *Overview* → *Download publish profile*. In your GitHub repo → *Settings* →
   *Secrets and variables* → *Actions*:
   - *Secrets* → New secret `AZURE_WEBAPP_PUBLISH_PROFILE` = the whole contents of the downloaded file.
   - *Variables* → New variable `AZURE_WEBAPP_NAME` = your Web App name.
5. **Deploy:** push to `main` (or *Actions* → *CI/CD* → *Run workflow*). The pipeline runs the tests, then deploys.
   The first start takes a few minutes while Azure installs the packages.

Free-tier note: F1 apps sleep when idle, so the first request after a while is slow. Brand kits are saved under `/home/data`.

## Project structure

```
app/
  main.py         FastAPI routes and static UI
  agent.py        LangGraph graph: retrieve → draft → review → revise
  brand_kits.py   Text extraction, chunking, FAISS index per brand kit
  llm.py          Gemini chat + embeddings, or offline fakes in mock mode
  config.py       Settings from environment variables / .env
  static/         Web UI (HTML + JavaScript)
samples/          Example brand guidelines
tests/            API tests (pytest, mock mode)
.github/workflows CI (tests) + CD (Azure deploy)
```

## Tech stack

Python · FastAPI · LangChain · LangGraph · Gemini API · FAISS · JavaScript · pytest · GitHub Actions · Azure App Service
