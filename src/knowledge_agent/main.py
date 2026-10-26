from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from knowledge_agent.config import get_settings
from knowledge_agent.schemas import AskRequest, AskResponse, IngestResponse
from knowledge_agent.service import get_agent, get_index

app = FastAPI(
    title="Enterprise Knowledge Intelligence Agent",
    description="A citation-grounded RAG API powered by Groq, LangGraph, and FAISS.",
    version="0.1.0",
)


@app.get("/health", tags=["operations"])
def health():
    return {"status": "healthy", "service": "enterprise-knowledge-intelligence-agent"}


@app.post("/ask", response_model=AskResponse, tags=["knowledge"])
def ask(request: AskRequest, agent=Depends(get_agent)):
    try:
        return agent.ask(request.question)
    except Exception as exc:
        # Keep implementation details and secrets out of client responses.
        raise HTTPException(status_code=502, detail="Knowledge query failed. Check server logs.") from exc


@app.post("/ingest", response_model=IngestResponse, tags=["knowledge"])
async def ingest(file: UploadFile = File(...), index=Depends(get_index)):
    settings = get_settings()
    filename = Path(file.filename or "upload.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=415, detail="Only .txt, .md, and .pdf files are supported.")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / filename
    try:
        target.write_bytes(content)
        count = index.ingest_path(target)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Document ingestion failed. Check server logs.") from exc
    return IngestResponse(filename=filename, chunks_indexed=count, message="Document indexed successfully.")
