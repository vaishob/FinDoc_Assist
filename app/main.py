from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .db import db
from .dependencies import get_ingestion_service, get_query_service
from .schemas import (
    DocumentDetailResponse,
    DocumentItem,
    DocumentListResponse,
    ErrorResponse,
    QueryRequest,
    SummaryResponse,
    UploadResponse,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def document_row_to_schema(row) -> DocumentItem:
    return DocumentItem(
        document_id=row["id"],
        filename=row["filename"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        pii_detected=bool(row["pii_detected"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = request.headers.get("x-request-id")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=str(exc.status_code),
            message=str(exc.detail),
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    documents = [document_row_to_schema(row) for row in db.list_documents()]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "documents": documents,
        },
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    ingestion_service = get_ingestion_service()
    document_id = await ingestion_service.save_upload(file)
    background_tasks.add_task(ingestion_service.process_document, document_id)
    return UploadResponse(
        document_id=document_id,
        status="pending",
        message="Upload accepted for processing",
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    return DocumentListResponse(documents=[document_row_to_schema(row) for row in db.list_documents()])


@app.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str):
    row = db.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    base = document_row_to_schema(row)
    return DocumentDetailResponse(
        **base.model_dump(),
        summary=row["summary"],
        error_message=row["error_message"],
    )


@app.get("/documents/{document_id}/summary", response_model=SummaryResponse)
async def get_summary(document_id: str):
    row = db.get_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return SummaryResponse(document_id=document_id, summary=row["summary"])


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    ingestion_service = get_ingestion_service()
    if not ingestion_service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "status": "deleted"}


@app.post("/query")
async def query_documents(payload: QueryRequest):
    if payload.document_ids:
        for document_id in payload.document_ids:
            if db.get_document(document_id) is None:
                raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    query_service = get_query_service()
    try:
        return await query_service.answer(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream inference failure: {exc}") from exc


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
