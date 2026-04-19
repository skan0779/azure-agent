import type { FastifyPluginAsync } from "fastify";

import type { ThreadRepository } from "../lib/thread-repository.js";
import {
  deleteThreadRoute,
  getThreadMessagesRoute,
  listThreadsRoute,
  updateThreadRoute,
} from "./threads.contract.js";

const resolveUserId = (headerUserId: unknown) => {
  return typeof headerUserId === "string" && headerUserId.trim()
    ? headerUserId.trim()
    : null;
};

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

      const userId = resolveUserId(request.headers["x-user-id"]);
      if (!userId) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: "Missing X-User-Id header",
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

      const userId = resolveUserId(request.headers["x-user-id"]);
      if (!userId) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: "Missing X-User-Id header",
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

      const userId = resolveUserId(request.headers["x-user-id"]);
      if (!userId) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: "Missing X-User-Id header",
        };
      }
      const currentThreads = await threadRepository.listThreadsForUser(userId);
      const existing = currentThreads.find(
        (thread) => thread.id === parsedParams.data.threadId,
      );

      if (!existing) {
        reply.code(404);
        return {
          error: "not_found",
          detail: "Thread not found",
        };
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

      const userId = resolveUserId(request.headers["x-user-id"]);
      if (!userId) {
        reply.code(400);
        return {
          error: "invalid_request",
          detail: "Missing X-User-Id header",
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
