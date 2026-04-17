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
  buildCitationParts,
  buildDynamicToolParts,
  collectToolSnapshotsFromUpdate,
  createAutoThreadTitle,
  extractLangChainChunkText,
  extractLastUserMessage,
  extractLastUserText,
  resolveThreadId,
} from "../lib/chat.js";
import { config } from "../config.js";
import type { ThreadRepository } from "../lib/thread-repository.js";
import type { UIMessage } from "../lib/thread-history.js";

const chatBodySchema = z.object({
  id: z.string().optional(),
  messages: z.array(z.any()).default([]),
  threadId: z.string().uuid().optional(),
  userId: z.string().min(1).optional(),
  trigger: z.string().optional(),
});

const cancelBodySchema = z.object({
  jobId: z.string().min(1),
  userId: z.string().min(1).optional(),
});

const COMPLETED_CANCEL_TTL_MS = 30_000;
const inFlightJobCancels = new Map<string, Promise<void>>();
const completedJobCancels = new Map<string, number>();

const isAbortError = (error: unknown): boolean => {
  return (
    error instanceof Error &&
    (error.name === "AbortError" ||
      error.message.toLowerCase().includes("aborted"))
  );
};

const buildStreamCorsHeaders = (
  origin: string | undefined,
): Record<string, string> => {
  if (!origin) {
    return {};
  }

  if (!config.isCorsOriginAllowed(origin)) {
    return {};
  }

  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Credentials": "true",
    Vary: "Origin",
  };
};

const cleanupCompletedJobCancels = () => {
  const now = Date.now();

  for (const [jobId, expiresAt] of completedJobCancels.entries()) {
    if (expiresAt <= now) {
      completedJobCancels.delete(jobId);
    }
  }
};

const runJobCancel = async ({
  jobId,
  cancel,
}: {
  jobId: string;
  cancel: () => Promise<void>;
}) => {
  cleanupCompletedJobCancels();

  const completedAt = completedJobCancels.get(jobId);
  if (completedAt && completedAt > Date.now()) {
    return;
  }

  const inFlight = inFlightJobCancels.get(jobId);
  if (inFlight) {
    await inFlight;
    return;
  }

  const cancelPromise = (async () => {
    try {
      await cancel();
      completedJobCancels.set(jobId, Date.now() + COMPLETED_CANCEL_TTL_MS);
    } finally {
      inFlightJobCancels.delete(jobId);
    }
  })();

  inFlightJobCancels.set(jobId, cancelPromise);
  await cancelPromise;
};

export const buildChatRoutes = ({
  threadRepository,
}: {
  threadRepository: ThreadRepository | null;
}): FastifyPluginAsync => {
  const chatRoutes: FastifyPluginAsync = async (app) => {
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

    app.log.info(
      {
        jobId: parsed.data.jobId,
        userId: resolvedUserId,
        agentApiBaseUrl: config.agentApiBaseUrl,
      },
      "Cancelling agent job",
    );

    await runJobCancel({
      jobId: parsed.data.jobId,
      cancel: () =>
        cancelAgentJobById({
          baseUrl: config.agentApiBaseUrl,
          jobId: parsed.data.jobId,
          userId: resolvedUserId,
        }),
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
    const lastUserMessage = extractLastUserMessage(messages);
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
    const resolvedThreadId = resolveThreadId(threadId, id);
    const abortController = new AbortController();
    let activeJob: AgentJobCreateResponse | null = null;
    let isTerminal = false;
    let cancelRequested = false;
    let assistantMessageId: string | null = null;
    let assistantText = "";
    const toolSnapshots = new Map<
      string,
      {
        toolName: string;
        toolCallId: string;
        input?: unknown;
        output?: unknown;
        errorText?: string;
        state: "input-available" | "output-available" | "output-error";
      }
    >();

    const requestJobCancel = async () => {
      if (cancelRequested || isTerminal || !activeJob) {
        return;
      }

      cancelRequested = true;
      const job = activeJob;

      try {
        await runJobCancel({
          jobId: job.job_id,
          cancel: () =>
            cancelAgentJob({
              cancelUrl: job.cancel_url,
              userId: resolvedUserId,
            }),
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

    request.raw.once("aborted", abortHandler);

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        try {
          if (threadRepository) {
            await threadRepository.upsertThread({
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              title: createAutoThreadTitle(lastUserText),
              updatedAt: new Date().toISOString(),
              titleSource: "first-user-message",
            });

            await threadRepository.upsertMessage({
              threadId: resolvedThreadId,
              message:
                lastUserMessage ??
                ({
                  id: crypto.randomUUID(),
                  role: "user",
                  parts: [
                    {
                      type: "text",
                      text: lastUserText,
                    },
                  ],
                } satisfies UIMessage),
            });
          }

          app.log.info(
            {
              requestId: request.id,
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              messageCount: messages.length,
              lastUserText,
              agentApiBaseUrl: config.agentApiBaseUrl,
            },
            "Starting chat request",
          );

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

          app.log.info(
            {
              requestId: request.id,
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              agentApiBaseUrl: config.agentApiBaseUrl,
            },
            "Creating agent job",
          );

          const job = await createAgentJob({
            baseUrl: config.agentApiBaseUrl,
            threadId: resolvedThreadId,
            userId: resolvedUserId,
            userQuery: lastUserText,
            signal: abortController.signal,
          });
          activeJob = job;

          if (threadRepository) {
            await threadRepository.upsertThread({
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              title: createAutoThreadTitle(lastUserText),
              updatedAt: new Date().toISOString(),
              lastJobId: job.job_id,
              titleSource: "first-user-message",
            });
          }

          app.log.info(
            {
              requestId: request.id,
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              jobId: job.job_id,
              status: job.status,
              eventsUrl: job.events_url,
            },
            "Created agent job",
          );

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

          app.log.info(
            {
              requestId: request.id,
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              jobId: job.job_id,
            },
            "Opening agent event stream",
          );

          for await (const event of streamAgentEvents({
            eventsUrl: job.events_url,
            userId: resolvedUserId,
            signal: abortController.signal,
          })) {
            if (event.type === "messages" && Array.isArray(event.data)) {
              const message = event.data[0];
              const metadata = event.data[1];
              const delta = extractLangChainChunkText(message, metadata);
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
                app.log.info(
                  {
                    requestId: request.id,
                    threadId: resolvedThreadId,
                    userId: resolvedUserId,
                    jobId: job.job_id,
                  },
                  "Received first text chunk from agent stream",
                );
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
              assistantText += delta;
              writer.write({
                type: "text-delta",
                id: textId,
                delta,
              });
              continue;
            }

            if (event.type === "complete") {
              isTerminal = true;
              app.log.info(
                {
                  requestId: request.id,
                  threadId: resolvedThreadId,
                  userId: resolvedUserId,
                  jobId: job.job_id,
                },
                "Agent stream completed",
              );
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
              app.log.info(
                {
                  requestId: request.id,
                  threadId: resolvedThreadId,
                  userId: resolvedUserId,
                  jobId: job.job_id,
                },
                "Agent stream cancelled",
              );
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

            if (event.type === "updates") {
              const toolChunks = collectToolSnapshotsFromUpdate(
                event.data,
                toolSnapshots,
              );
              for (const chunk of toolChunks) {
                writer.write(chunk);
              }
            }
          }

          isTerminal = true;

          if (textId) {
            writer.write({
              type: "text-end",
              id: textId,
            });

            if (threadRepository && (assistantText.trim() || toolSnapshots.size > 0)) {
              assistantMessageId ??= crypto.randomUUID();
              const toolParts = buildDynamicToolParts(toolSnapshots);
              const citationParts = buildCitationParts(toolSnapshots);
              await threadRepository.upsertMessage({
                threadId: resolvedThreadId,
                message: {
                  id: assistantMessageId,
                  role: "assistant",
                  metadata:
                    citationParts.length > 0
                      ? {
                          citations: citationParts,
                        }
                      : undefined,
                  parts: [
                    ...toolParts,
                    ...(assistantText.trim()
                      ? [
                          {
                            type: "text" as const,
                            text: assistantText,
                            state: "done" as const,
                          },
                        ]
                      : []),
                  ],
                },
              });
            }
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
            app.log.info(
              {
                requestId: request.id,
                threadId: resolvedThreadId,
                userId: resolvedUserId,
                jobId: activeJob?.job_id,
              },
              "Chat request aborted by client",
            );
            await requestJobCancel();
            return;
          }

          app.log.error(
            {
              err: error,
              requestId: request.id,
              threadId: resolvedThreadId,
              userId: resolvedUserId,
              jobId: activeJob?.job_id,
              agentApiBaseUrl: config.agentApiBaseUrl,
            },
            "Chat request failed",
          );

          throw error;
        }
      },
    });

    reply.hijack();
    reply.raw.once("close", () => {
      if (reply.raw.writableEnded || isTerminal) {
        return;
      }

      abortHandler();
    });
    pipeUIMessageStreamToResponse({
      response: reply.raw,
      stream,
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
        ...buildStreamCorsHeaders(request.headers.origin),
      },
    });

    request.raw.once("close", () => {
      request.raw.off("aborted", abortHandler);
    });
  });

  };

  return chatRoutes;
};
