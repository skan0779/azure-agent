from __future__ import annotations

import base64

from deepagents.backends.protocol import ReadResult
from deepagents.backends.utils import _get_file_type
from langchain_azure_dynamic_sessions.backends import SessionsBashBackend


class MultimodalSessionsBashBackend(SessionsBashBackend):
    """Azure Dynamic Sessions backend with DeepAgents multimodal reads.

    The upstream SessionsBashBackend overrides read() with an awk-based text
    reader. That is fine for text, but it prevents DeepAgents read_file from
    returning image/PDF/audio/video content blocks to the model. For non-text
    files, download the bytes from /mnt/data and return base64 FileData so
    DeepAgents FilesystemMiddleware can build the multimodal ToolMessage.
    """

    max_multimodal_bytes = 20 * 1024 * 1024

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult | str:
        if _get_file_type(file_path) == "text":
            return super().read(file_path, offset=offset, limit=limit)

        response = self.download_files([file_path])[0]
        if response.error is not None or response.content is None:
            return ReadResult(error=f"File '{file_path}' not found or unavailable")

        if len(response.content) > self.max_multimodal_bytes:
            return ReadResult(
                error=(
                    f"File '{file_path}' is too large for multimodal read "
                    f"({len(response.content)} bytes > {self.max_multimodal_bytes} bytes)"
                )
            )

        return ReadResult(
            file_data={
                "content": base64.b64encode(response.content).decode("ascii"),
                "encoding": "base64",
            }
        )
