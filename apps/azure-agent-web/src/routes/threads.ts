import type { FastifyPluginAsync } from "fastify";

import { getRequestUserId } from "../auth.js";
import { deleteAgentThreadFiles } from "../lib/azure-agent-api.js";
import { config } from "../config.js";
import type { ThreadRepository } from "../lib/thread-repository.js";
import {
  deleteThreadRoute,
  getThreadMessagesRoute,
  listThreadsRoute,
  updateThreadRoute,
} from "./threads.contract.js";

export const buildThreadsRoutes = ({
  threadRepository,
}: {
  threadRepository: ThreadRepository | null;
}): FastifyPluginAsync => {
  const threadsRoutes: FastifyPluginAsync = async (app) => {
    app.get(listThreadsRoute.url, async (request, reply) => {
      if (!threadRepository) {
        reply.code(503);
        return {
          error: "service_unavailable",
          detail: "Thread storage is not configured",
        };
      }

      const userId = getRequestUserId(request);
      if (!userId) {
        reply.code(401);
        return {
          error: "missing_user_identity",
          detail: "Authenticated user is required",
        };
      }
      const threads = await threadRepository.listThreadsForUser(userId);
      return threads;
    });

    app.get(getThreadMessagesRoute.url, async (request, reply) => {
      if (!threadRepository) {
        reply.code(503);
        return {
          error: "service_unavailable",
          detail: "Thread storage is not configured",
        };
      }

      const parsed = getThreadMessagesRoute.paramsSchema.safeParse(
        request.params,
      );
      if (!parsed.success) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: parsed.error.flatten(),
        };
      }

      const userId = getRequestUserId(request);
      if (!userId) {
        reply.code(401);
        return {
          error: "missing_user_identity",
          detail: "Authenticated user is required",
        };
      }
      const messages = await threadRepository.getThreadMessages({
        threadId: parsed.data.threadId,
        userId,
      });

      return messages;
    });

    app.patch(updateThreadRoute.url, async (request, reply) => {
      if (!threadRepository) {
        reply.code(503);
        return {
          error: "service_unavailable",
          detail: "Thread storage is not configured",
        };
      }

      const parsedParams = updateThreadRoute.paramsSchema.safeParse(
        request.params,
      );
      if (!parsedParams.success) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: parsedParams.error.flatten(),
        };
      }

      const parsedBody = updateThreadRoute.bodySchema.safeParse(request.body);
      if (!parsedBody.success) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: parsedBody.error.flatten(),
        };
      }

      const userId = getRequestUserId(request);
      if (!userId) {
        reply.code(401);
        return {
          error: "missing_user_identity",
          detail: "Authenticated user is required",
        };
      }
      const currentThreads = await threadRepository.listThreadsForUser(userId);
      const existing = currentThreads.find(
        (thread) => thread.id === parsedParams.data.threadId,
      );

      if (!existing) {
        return threadRepository.upsertThread({
          threadId: parsedParams.data.threadId,
          userId,
          title: parsedBody.data.title?.trim() || "New chat",
          updatedAt: parsedBody.data.updatedAt ?? new Date().toISOString(),
          lastJobId: parsedBody.data.lastJobId,
          titleSource: parsedBody.data.titleSource,
        });
      }

      return threadRepository.upsertThread({
        threadId: existing.id,
        userId,
        title: parsedBody.data.title?.trim() || existing.title,
        updatedAt: parsedBody.data.updatedAt ?? new Date().toISOString(),
        lastJobId: parsedBody.data.lastJobId ?? existing.lastJobId,
        titleSource:
          parsedBody.data.title?.trim() === existing.title
            ? existing.titleSource
            : (parsedBody.data.titleSource ?? existing.titleSource),
      });
    });

    app.delete(deleteThreadRoute.url, async (request, reply) => {
      if (!threadRepository) {
        reply.code(503);
        return {
          error: "service_unavailable",
          detail: "Thread storage is not configured",
        };
      }

      const parsed = deleteThreadRoute.paramsSchema.safeParse(request.params);
      if (!parsed.success) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: parsed.error.flatten(),
        };
      }

      const userId = getRequestUserId(request);
      if (!userId) {
        reply.code(401);
        return {
          error: "missing_user_identity",
          detail: "Authenticated user is required",
        };
      }

      try {
        const cleanup = await deleteAgentThreadFiles({
          baseUrl: config.agentApiBaseUrl,
          threadId: parsed.data.threadId,
          userId,
        });
        request.log.info(
          {
            threadId: parsed.data.threadId,
            userId,
            deletedFiles: cleanup.deleted_files,
            deletedBlobs: cleanup.deleted_blobs,
          },
          "Deleted agent files for thread",
        );
      } catch (error) {
        request.log.error(
          {
            err: error,
            threadId: parsed.data.threadId,
            userId,
          },
          "Failed to delete agent files for thread; aborting thread deletion",
        );
        reply.code(502);
        return {
          error: "agent_cleanup_failed",
          detail:
            error instanceof Error
              ? error.message
              : "Failed to delete agent files",
        };
      }

      await threadRepository.deleteThread({
        threadId: parsed.data.threadId,
        userId,
      });

      reply.code(204);
      return null;
    });
  };

  return threadsRoutes;
};
