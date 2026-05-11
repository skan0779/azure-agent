import multipart from "@fastify/multipart";
import type { FastifyPluginAsync } from "fastify";
import { Readable } from "node:stream";

import {
  downloadAgentFile,
  uploadAgentFile,
} from "../lib/azure-agent-api.js";
import { config } from "../config.js";

const MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024;

const resolveUserId = (
  bodyUserId: string | undefined,
  headerUserId: string | undefined,
): string | null => {
  const resolved = bodyUserId?.trim() || headerUserId?.trim();
  return resolved || null;
};

export const filesRoutes: FastifyPluginAsync = async (app) => {
  await app.register(multipart, {
    limits: {
      fileSize: MAX_UPLOAD_SIZE_BYTES,
      files: 1,
    },
  });

  app.post("/api/files", async (request, reply) => {
    const headerUserId =
      typeof request.headers["x-user-id"] === "string"
        ? request.headers["x-user-id"]
        : undefined;

    const userId = resolveUserId(undefined, headerUserId);
    if (!userId) {
      reply.code(401);
      return {
        error: "missing_user_identity",
        detail: "X-User-Id header is required",
      };
    }

    let part;
    try {
      part = await request.file();
    } catch (error) {
      app.log.warn({ err: error }, "Failed to read multipart upload");
      reply.code(400);
      return {
        error: "invalid_request",
        detail: "Invalid multipart payload",
      };
    }

    if (!part) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: "Missing file part",
      };
    }

    const threadIdField = part.fields?.thread_id;
    const threadIdRaw =
      threadIdField &&
      !Array.isArray(threadIdField) &&
      "value" in threadIdField &&
      typeof threadIdField.value === "string"
        ? threadIdField.value.trim()
        : "";

    if (!threadIdRaw) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: "Missing thread_id field",
      };
    }

    let buffer: Buffer;
    try {
      buffer = await part.toBuffer();
    } catch (error) {
      app.log.warn({ err: error }, "Failed to buffer uploaded file");
      reply.code(413);
      return {
        error: "payload_too_large",
        detail: `File exceeds the ${MAX_UPLOAD_SIZE_BYTES} byte limit`,
      };
    }

    if (part.file.truncated) {
      reply.code(413);
      return {
        error: "payload_too_large",
        detail: `File exceeds the ${MAX_UPLOAD_SIZE_BYTES} byte limit`,
      };
    }

    try {
      const result = await uploadAgentFile({
        baseUrl: config.agentApiBaseUrl,
        threadId: threadIdRaw,
        userId,
        file: {
          buffer,
          filename: part.filename ?? "file",
          mimeType: part.mimetype,
        },
      });
      reply.code(201);
      return result;
    } catch (error) {
      app.log.error(
        { err: error, threadId: threadIdRaw, userId },
        "Failed to forward upload to agent API",
      );
      reply.code(502);
      return {
        error: "upstream_failure",
        detail:
          error instanceof Error ? error.message : "Failed to upload file",
      };
    }
  });

  app.get("/api/files/:fileId/download", async (request, reply) => {
    const headerUserId =
      typeof request.headers["x-user-id"] === "string"
        ? request.headers["x-user-id"]
        : undefined;

    const userId = resolveUserId(undefined, headerUserId);
    if (!userId) {
      reply.code(401);
      return {
        error: "missing_user_identity",
        detail: "X-User-Id header is required",
      };
    }

    const fileId =
      typeof (request.params as { fileId?: string }).fileId === "string"
        ? (request.params as { fileId: string }).fileId
        : "";
    if (!fileId) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: "Missing fileId parameter",
      };
    }

    let upstream: Response;
    try {
      upstream = await downloadAgentFile({
        baseUrl: config.agentApiBaseUrl,
        fileId,
        userId,
      });
    } catch (error) {
      app.log.error(
        { err: error, fileId, userId },
        "Failed to fetch file from agent API",
      );
      reply.code(502);
      return {
        error: "upstream_failure",
        detail:
          error instanceof Error ? error.message : "Failed to download file",
      };
    }

    const contentType =
      upstream.headers.get("content-type") ?? "application/octet-stream";
    const contentDisposition =
      upstream.headers.get("content-disposition") ?? `attachment; filename="${fileId}"`;
    const contentLength = upstream.headers.get("content-length");

    reply.code(upstream.status);
    reply.header("Content-Type", contentType);
    reply.header("Content-Disposition", contentDisposition);
    if (contentLength) {
      reply.header("Content-Length", contentLength);
    }

    if (!upstream.body) {
      return reply.send();
    }

    return reply.send(Readable.fromWeb(upstream.body as never));
  });
};
