# Enterprise Knowledge Intelligence Agent

A backend-only, citation-grounded enterprise RAG API using **FastAPI, LangGraph, Groq, FAISS, and sentence-transformers**. Built as a portfolio project with tests, a baseline evaluation harness, Docker, and GitHub Actions.

## Architecture

1. Upload `.txt`, `.md`, or text-based `.pdf` documents to `/ingest`.
2. Split text into overlapping chunks and embed them with a local sentence-transformers model.
3. Store vectors in FAISS and persist the index under `data/faiss_index/`.
4. Ask questions through `/ask`; LangGraph coordinates retrieval and grounded answer generation through Groq.
5. Responses include retrieved source names and excerpts as citations.

## Requirements

- Python 3.11
- A Groq API key for live generated answers
- Internet access for initial dependency and embedding-model downloads

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

Add your key to `.env` as `GROQ_API_KEY=your_key_here`. Never commit `.env`.

Start the API:

```powershell
uvicorn knowledge_agent.main:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/docs`. Health check: `http://127.0.0.1:8000/health`.

## API examples

Health:

```bash
curl http://127.0.0.1:8000/health
```

Ingest a document:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@data/sample_knowledge.md"
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What does RAG stand for?\"}"
```

## Tests and evaluation

```bash
pytest
ruff check src tests evaluation
python evaluation/evaluate.py evaluation/results.example.json
```

The evaluation script uses transparent keyword and source checks. These are only a baseline; do not present them as a comprehensive hallucination or semantic-quality benchmark.

## Docker

```bash
docker compose up --build
```

Create `.env` first. The local `data` directory is mounted into the container so the FAISS index persists.

## Security notes and current limitations

- Secrets are read from environment configuration; `.env` is excluded from Git.
- Upload file extensions and size are checked, but this is a learning/portfolio baseline, not a hardened multi-tenant enterprise service.
- No authentication, authorization, rate limiting, tenant isolation, malware scanning, or audit log is implemented yet. Add these before production use.
- Only `.txt`, `.md`, and text-based PDFs are supported; scanned PDFs need OCR.
- Local embedding models are downloaded on first use.
- Treat retrieved documents as untrusted input. Prompt instructions reduce but do not eliminate prompt-injection risk.
- Citation excerpts are retrieved evidence, not a guarantee that every generated claim is supported.
- FAISS persistence uses a local file index; keep the index directory trusted and backed up.

## Repository checklist

- [ ] Add a real Groq key to local `.env` only.
- [ ] Run tests and lint locally.
- [ ] Test `/ingest` and `/ask` with your own sample file.
- [ ] Confirm no secrets or private documents are staged.
- [ ] Push to GitHub and confirm Actions passes.
- [ ] Add screenshots of `/docs`, test results, and an architecture diagram to the repository.

## License

Choose a license before public release (MIT is a common option for portfolio projects).
