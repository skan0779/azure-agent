import { createUIMessageStream, pipeUIMessageStreamToResponse } from "ai";
import type { FastifyPluginAsync } from "fastify";
import { z } from "zod";

import {
  cancelAgentJob,
  cancelAgentJobById,
  createAgentJob,
  type AgentJobCreateResponse,
  streamAgentEvents,
} from "../lib/azure-agent-api.js";
import {
  extractLangChainChunkText,
  extractLastUserText,
  resolveThreadId,
} from "../lib/chat.js";
import { config } from "../config.js";

const chatBodySchema = z.object({
  id: z.string().optional(),
  messages: z.array(z.any()).default([]),
  threadId: z.string().optional(),
  userId: z.string().min(1).optional(),
  trigger: z.string().optional(),
});

const cancelBodySchema = z.object({
  jobId: z.string().min(1),
  userId: z.string().min(1).optional(),
});

const isAbortError = (error: unknown): boolean => {
  return (
    error instanceof Error &&
    (error.name === "AbortError" ||
      error.message.toLowerCase().includes("aborted"))
  );
};

export const chatRoutes: FastifyPluginAsync = async (app) => {
  app.post("/api/chat/cancel", async (request, reply) => {
    const parsed = cancelBodySchema.safeParse(request.body);
    if (!parsed.success) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: parsed.error.flatten(),
      };
    }

    const headerUserId =
      typeof request.headers["x-user-id"] === "string"
        ? request.headers["x-user-id"]
        : undefined;
    const resolvedUserId =
      parsed.data.userId?.trim() ||
      headerUserId?.trim() ||
      config.defaultUserId;

    await cancelAgentJobById({
      baseUrl: config.agentApiBaseUrl,
      jobId: parsed.data.jobId,
      userId: resolvedUserId,
    });

    return {
      ok: true,
      jobId: parsed.data.jobId,
      userId: resolvedUserId,
    };
  });

  app.post("/api/chat", async (request, reply) => {
    const parsed = chatBodySchema.safeParse(request.body);
    if (!parsed.success) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: parsed.error.flatten(),
      };
    }

    const { id, messages, threadId, userId } = parsed.data;
    const lastUserText = extractLastUserText(messages);
    if (!lastUserText) {
      reply.code(400);
      return {
        error: "invalid_request",
        detail: "Unable to find the latest user message in the request payload",
      };
    }

    const headerUserId =
      typeof request.headers["x-user-id"] === "string"
        ? request.headers["x-user-id"]
        : undefined;
    const resolvedUserId =
      userId?.trim() || headerUserId?.trim() || config.defaultUserId;
    const resolvedThreadId = resolveThreadId(threadId, id, resolvedUserId);
    const abortController = new AbortController();
    let activeJob: AgentJobCreateResponse | null = null;
    let isTerminal = false;
    let cancelRequested = false;

    const requestJobCancel = async () => {
      if (cancelRequested || isTerminal || !activeJob) {
        return;
      }

      cancelRequested = true;

      try {
        await cancelAgentJob({
          cancelUrl: activeJob.cancel_url,
          userId: resolvedUserId,
        });
      } catch (error) {
        app.log.warn(
          {
            err: error,
            jobId: activeJob.job_id,
            threadId: resolvedThreadId,
          },
          "Failed to cancel agent job after client abort",
        );
      }
    };

    const abortHandler = () => {
      if (isTerminal) {
        return;
      }

      abortController.abort();
      void requestJobCancel();
    };

    request.raw.once("close", abortHandler);

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        try {
          writer.write({
            type: "data-metadata",
            data: {
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              chatId: id ?? null,
            },
          });

          writer.write({
            type: "data-status",
            data: {
              stage: "submitting",
              threadId: resolvedThreadId,
              userId: resolvedUserId,
            },
          });

          const job = await createAgentJob({
            baseUrl: config.agentApiBaseUrl,
            threadId: resolvedThreadId,
            userId: resolvedUserId,
            userQuery: lastUserText,
            signal: abortController.signal,
          });
          activeJob = job;

          writer.write({
            type: "data-status",
            data: {
              stage: "queued",
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              jobId: job.job_id,
              status: job.status,
            },
          });

          writer.write({
            type: "data-metadata",
            data: {
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              chatId: id ?? null,
              jobId: job.job_id,
            },
          });

          let textId: string | undefined;
          let sawText = false;
          let emittedStreamingStatus = false;

          for await (const event of streamAgentEvents({
            eventsUrl: job.events_url,
            userId: resolvedUserId,
            signal: abortController.signal,
          })) {
            if (event.type === "messages" && Array.isArray(event.data)) {
              const message = event.data[0];
              const delta = extractLangChainChunkText(message);
              if (!delta) {
                continue;
              }

              if (!textId) {
                textId = crypto.randomUUID();
                writer.write({
                  type: "text-start",
                  id: textId,
                });
              }

              if (!emittedStreamingStatus) {
                emittedStreamingStatus = true;
                writer.write({
                  type: "data-status",
                  data: {
                    stage: "streaming",
                    threadId: resolvedThreadId,
                    userId: resolvedUserId,
                    jobId: job.job_id,
                  },
                });
              }

              sawText = true;
              writer.write({
                type: "text-delta",
                id: textId,
                delta,
              });
              continue;
            }

            if (event.type === "complete") {
              isTerminal = true;
              break;
            }

            if (event.type === "error") {
              const message =
                event.data &&
                typeof event.data === "object" &&
                "message" in event.data &&
                typeof event.data.message === "string"
                  ? event.data.message
                  : "Agent stream error";
              throw new Error(message);
            }

            if (event.type === "cancelled") {
              isTerminal = true;
              writer.write({
                type: "data-status",
                data: {
                  stage: "cancelled",
                  threadId: resolvedThreadId,
                  userId: resolvedUserId,
                  jobId: job.job_id,
                },
              });
              break;
            }

            writer.write({
              type: "data-agent-event",
              data: {
                jobId: job.job_id,
                threadId: resolvedThreadId,
                userId: resolvedUserId,
                event,
              },
            });
          }

          isTerminal = true;

          if (textId) {
            writer.write({
              type: "text-end",
              id: textId,
            });
          } else if (!sawText) {
            writer.write({
              type: "data-status",
              data: {
                stage: "complete",
                threadId: resolvedThreadId,
                userId: resolvedUserId,
                jobId: job.job_id,
                note: "No assistant text chunks were emitted",
              },
            });
          } else {
            writer.write({
              type: "data-status",
              data: {
                stage: "complete",
                threadId: resolvedThreadId,
                userId: resolvedUserId,
                jobId: job.job_id,
              },
            });
          }
        } catch (error) {
          if (isAbortError(error) || abortController.signal.aborted) {
            await requestJobCancel();
            return;
          }

          throw error;
        }
      },
    });

    reply.hijack();
    pipeUIMessageStreamToResponse({
      response: reply.raw,
      stream,
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
      },
    });

    request.raw.once("close", () => {
      request.raw.off("close", abortHandler);
    });
  });
};
