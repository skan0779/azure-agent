import logging
import re
from typing import Annotated
from urllib.parse import quote
from uuid import UUID, uuid4

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import ContainerClient
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse

from azure_agent.api.schema import ErrorResponse, FileRole, FileUploadResponse
from azure_agent.files import AgentFileCreate, AgentFileRepository

logger = logging.getLogger(__name__)

router = APIRouter()


def sanitize_sandbox_filename(filename: str, *, max_length: int = 120) -> str:
    filename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename)
    filename = filename.strip(" .")
    return filename[:max_length] or "file"


def get_request_user_id(
    x_user_id: Annotated[
        str | None,
        Header(
            alias="X-User-Id",
            description="Caller identity used for file ownership checks.",
            examples=["user-123"],
        ),
    ] = None,
) -> str:
    user_id = str(x_user_id or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "missing_user_identity",
                "message": "X-User-Id header is required",
            },
        )
    return user_id


def get_file_repository(request: Request) -> AgentFileRepository:
    repository = getattr(request.app.state, "agent_file_repository", None)
    if repository is None:
        raise HTTPException(status_code=500, detail="File repository unavailable")
    return repository


def get_blob_container_client(request: Request) -> ContainerClient:
    container_client = getattr(request.app.state, "blob_container_client", None)
    if container_client is None:
        raise HTTPException(status_code=500, detail="Blob container client unavailable")
    return container_client


@router.post(
    "/agent/api/files",
    response_model=FileUploadResponse,
    status_code=201,
    tags=["Files"],
    summary="Upload file",
    description="Uploads a user file to Blob Storage and records file metadata for a thread.",
    response_description="Uploaded file metadata.",
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Missing or empty `X-User-Id` header.",
        },
        500: {
            "model": ErrorResponse,
            "description": "The API could not store the file or metadata.",
        },
    },
)
async def upload_file_endpoint(
    thread_id: Annotated[UUID, Form(description="thread/session ID")],
    file: Annotated[UploadFile, File(description="file to upload")],
    request_user_id: Annotated[str, Depends(get_request_user_id)],
    repository: Annotated[AgentFileRepository, Depends(get_file_repository)],
    container_client: Annotated[ContainerClient, Depends(get_blob_container_client)],
) -> FileUploadResponse:
    file_id = str(uuid4())
    thread_id_str = str(thread_id)
    raw_filename = file.filename or file_id
    sandbox_filename = sanitize_sandbox_filename(raw_filename)
    filename = await repository.resolve_filename_collision(
        user_id=request_user_id,
        thread_id=thread_id_str,
        role="upload",
        filename=sandbox_filename,
    )
    blob_path = f"{request_user_id}/{thread_id_str}/uploads/{file_id}"
    sandbox_path = f"/mnt/data/{filename}"
    blob_uploaded = False

    try:
        content = await file.read()
        blob_client = container_client.get_blob_client(blob_path)
        await blob_client.upload_blob(
            content,
            overwrite=False,
            content_settings=ContentSettings(content_type=file.content_type),
        )
        blob_uploaded = True

        stored_file = await repository.insert_file_metadata(
            AgentFileCreate(
                file_id=file_id,
                user_id=request_user_id,
                thread_id=thread_id_str,
                job_id=None,
                role="upload",
                blob_path=blob_path,
                sandbox_path=sandbox_path,
                filename=filename,
                mime_type=file.content_type,
                size=len(content),
            )
        )
    except AzureError as exc:
        logger.exception("[files.py] Blob upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload file") from exc
    except Exception as exc:
        if blob_uploaded:
            try:
                await container_client.delete_blob(blob_path)
            except Exception:
                logger.warning("[files.py] Failed to delete orphan blob: %s", blob_path)
        logger.exception("[files.py] File metadata insert failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to save file metadata"
        ) from exc
    finally:
        await file.close()

    return FileUploadResponse(
        file_id=UUID(stored_file.file_id),
        thread_id=UUID(stored_file.thread_id),
        role=FileRole(stored_file.role),
        filename=stored_file.filename,
        mime_type=stored_file.mime_type,
        size=stored_file.size,
        created_at=stored_file.created_at,
    )


@router.get(
    "/agent/api/files/{file_id}/download",
    tags=["Files"],
    summary="Download file",
    description="Streams the binary content of a stored file (upload or artifact) for the caller.",
    response_description="Binary file content with original filename and MIME type.",
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Missing or empty `X-User-Id` header.",
        },
        404: {
            "model": ErrorResponse,
            "description": "File not found or not owned by the caller.",
        },
        500: {
            "model": ErrorResponse,
            "description": "The API could not stream the file from blob storage.",
        },
    },
)
async def download_file_endpoint(
    file_id: UUID,
    request_user_id: Annotated[str, Depends(get_request_user_id)],
    repository: Annotated[AgentFileRepository, Depends(get_file_repository)],
    container_client: Annotated[ContainerClient, Depends(get_blob_container_client)],
) -> StreamingResponse:
    stored_file = await repository.get_file(
        file_id=str(file_id),
        user_id=request_user_id,
    )
    if stored_file is None:
        raise HTTPException(status_code=404, detail="File not found")

    blob_client = container_client.get_blob_client(stored_file.blob_path)
    try:
        downloader = await blob_client.download_blob()
    except ResourceNotFoundError as exc:
        logger.warning(
            "[files.py] Blob missing for file_id=%s blob_path=%s",
            stored_file.file_id,
            stored_file.blob_path,
        )
        raise HTTPException(status_code=404, detail="File content not found") from exc
    except AzureError as exc:
        logger.exception("[files.py] Blob download failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to download file") from exc

    media_type = stored_file.mime_type or "application/octet-stream"
    encoded_filename = quote(stored_file.filename, safe="")
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{stored_file.filename}"; '
            f"filename*=UTF-8''{encoded_filename}"
        ),
    }
    if stored_file.size is not None:
        headers["Content-Length"] = str(stored_file.size)

    return StreamingResponse(
        downloader.chunks(),
        media_type=media_type,
        headers=headers,
    )


@router.delete(
    "/agent/api/threads/{thread_id}/files",
    status_code=200,
    tags=["Files"],
    summary="Delete all files for a thread",
    description=(
        "Removes every uploaded file and generated artifact stored for the given "
        "thread. Deletes Blob Storage objects, agent_files rows, and the "
        "sandbox_sessions row owned by the caller."
    ),
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Missing or empty `X-User-Id` header.",
        },
        500: {
            "model": ErrorResponse,
            "description": "The API could not delete files or metadata.",
        },
    },
)
async def delete_thread_files_endpoint(
    thread_id: UUID,
    request_user_id: Annotated[str, Depends(get_request_user_id)],
    repository: Annotated[AgentFileRepository, Depends(get_file_repository)],
    container_client: Annotated[ContainerClient, Depends(get_blob_container_client)],
) -> dict[str, int]:
    thread_id_str = str(thread_id)

    blob_paths = await repository.list_thread_blob_paths(
        user_id=request_user_id,
        thread_id=thread_id_str,
    )

    blob_failures = 0
    for blob_path in blob_paths:
        try:
            await container_client.delete_blob(blob_path)
        except ResourceNotFoundError:
            continue
        except AzureError as exc:
            blob_failures += 1
            logger.warning(
                "[files.py] Failed to delete blob during thread cleanup: %s (%s)",
                blob_path,
                exc,
            )

    if blob_failures > 0:
        # Some blobs could not be removed — keep DB rows so a later retry can
        # reconcile, rather than orphaning the blobs permanently.
        raise HTTPException(
            status_code=500,
            detail={
                "code": "blob_cleanup_partial_failure",
                "message": (
                    f"Failed to delete {blob_failures} of {len(blob_paths)} blob(s); "
                    "thread metadata retained for retry."
                ),
            },
        )

    deleted = await repository.delete_thread_metadata(
        user_id=request_user_id,
        thread_id=thread_id_str,
    )

    logger.info(
        "[files.py] Deleted thread files: thread_id=%s files=%d blobs=%d",
        thread_id_str,
        deleted,
        len(blob_paths),
    )

    return {
        "deleted_files": deleted,
        "deleted_blobs": len(blob_paths),
    }

