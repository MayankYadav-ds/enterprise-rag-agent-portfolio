"""Bounded raw-PDF uploads, durable status, inventory and managed deletion.

Private read routes require authentication. State-changing routes require
authentication and CSRF validation. Document recovery also requires admin
authorization.
"""

import os
import sqlite3
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from rag_agent.api.documents import (
    DocumentError,
    DocumentManager,
    DocumentRecord,
    safe_filename,
)
from rag_agent.api.runtime import DependencyUnavailable
from rag_agent.api.storage import private_file
from rag_agent.api.upload_rate_limit import reserve_upload_attempt, upload_bucket

# Correct import path for auth dependencies
from rag_agent.auth.dependencies import (
    User,
    csrf_required,
    get_current_user,
    require_admin,
    tenant_id_for_user,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def manager(request: Request) -> DocumentManager:
    return await run_in_threadpool(request.app.state.runtime.documents)


@router.post(
    "",
    response_model=DocumentRecord,
    status_code=202,
    responses={
        429: {"description": "Upload attempt limit exceeded"},
        503: {"description": "Upload admission storage is unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
    dependencies=[Depends(csrf_required)],
)
async def upload(
    request: Request,
    background: BackgroundTasks,
    user: Annotated[User, Depends(get_current_user)],
    filename: str = Query(max_length=120),
) -> DocumentRecord:
    runtime = request.app.state.runtime
    tenant_id = tenant_id_for_user(user)
    try:
        with runtime.connection() as connection:
            admission = reserve_upload_attempt(
                connection,
                upload_bucket(request.client.host if request.client else None),
                window_seconds=runtime.settings.upload_rate_limit_window_seconds,
                limit=runtime.settings.upload_rate_limit_attempts,
            )
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload admission temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc
    if not admission.admitted:
        raise HTTPException(
            status_code=429,
            detail="Upload rate limit exceeded",
            headers={"Retry-After": str(admission.retry_after or 1)},
        )
    filename = safe_filename(filename)
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/pdf":
        raise DocumentError(415, "invalid_media_type", "Upload raw PDF bytes with application/pdf")
    limit = runtime.settings.upload_max_bytes
    declared = request.headers.get("content-length")
    if declared:
        try:
            length = int(declared)
        except ValueError as exc:
            raise DocumentError(400, "invalid_length", "Invalid content length") from exc
        if length < 0 or length > limit:
            raise DocumentError(
                413, "file_too_large", "PDF exceeds the configured upload size limit"
            )
    service = await manager(request)
    path = service.tenant_root(tenant_id) / f"{uuid.uuid4().hex}.pdf"
    accepted = False
    try:
        # Exclusive creation under server-owned storage, never a client-selected path.
        size = 0
        header = bytearray()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            async for block in request.stream():
                size += len(block)
                if size > limit:
                    raise DocumentError(
                        413, "file_too_large", "PDF exceeds the configured upload size limit"
                    )
                if len(header) < 5:
                    header.extend(block[: 5 - len(header)])
                file.write(block)
        private_file(path)
        if header != b"%PDF-":
            raise DocumentError(415, "invalid_pdf", "The file does not have a PDF signature")
        try:
            record = await run_in_threadpool(service.accept, path, filename, tenant_id)
        except DocumentError:
            raise
        except Exception as exc:
            raise DependencyUnavailable("Document registration unavailable") from exc
        accepted = True
        background.add_task(service.process, record.doc_id, tenant_id)
        return record
    finally:
        if not accepted:
            path.unlink(missing_ok=True)


@router.get("", response_model=list[DocumentRecord])
async def inventory(
    request: Request, user: Annotated[User, Depends(get_current_user)]
) -> list[DocumentRecord]:
    service = await manager(request)
    try:
        return await run_in_threadpool(service.inventory, tenant_id_for_user(user))
    except Exception as exc:
        raise DependencyUnavailable("Document inventory unavailable") from exc


@router.post(
    "/recover",
    status_code=204,
    dependencies=[Depends(require_admin), Depends(csrf_required)],
)
async def recover(request: Request, user: Annotated[User, Depends(require_admin)]) -> None:
    service = await manager(request)
    tenant_id = tenant_id_for_user(user)
    await run_in_threadpool(service.recover, tenant_id)
    await run_in_threadpool(service.ensure_readable, tenant_id)


@router.get("/{doc_id}", response_model=DocumentRecord)
async def status(
    doc_id: str, request: Request, user: Annotated[User, Depends(get_current_user)]
) -> DocumentRecord:
    return await run_in_threadpool((await manager(request)).get, doc_id, tenant_id_for_user(user))


@router.delete(
    "/{doc_id}",
    response_model=DocumentRecord,
    dependencies=[Depends(csrf_required)],
)
async def delete(
    doc_id: str, request: Request, user: Annotated[User, Depends(get_current_user)]
) -> DocumentRecord:
    return await run_in_threadpool(
        (await manager(request)).delete, doc_id, tenant_id_for_user(user)
    )
