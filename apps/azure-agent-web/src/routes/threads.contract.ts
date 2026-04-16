import { z } from "zod";

import {
  listThreadMessagesResponseSchema,
  listThreadsResponseSchema,
  threadIdSchema,
  threadSummarySchema,
  threadTitleSourceSchema,
} from "../lib/thread-history.js";

export const listThreadsRoute = {
  method: "GET" as const,
  url: "/api/threads",
  responseSchema: listThreadsResponseSchema,
};

export const getThreadMessagesParamsSchema = z.object({
  threadId: threadIdSchema,
});

export const getThreadMessagesRoute = {
  method: "GET" as const,
  url: "/api/threads/:threadId/messages",
  paramsSchema: getThreadMessagesParamsSchema,
  responseSchema: listThreadMessagesResponseSchema,
};

export const updateThreadParamsSchema = z.object({
  threadId: threadIdSchema,
});

export const updateThreadBodySchema = z.object({
  title: z.string().min(1).optional(),
  titleSource: threadTitleSourceSchema.optional(),
  updatedAt: z.string().datetime({ offset: true }).optional(),
  lastJobId: z.string().min(1).optional(),
});

export const updateThreadRoute = {
  method: "PATCH" as const,
  url: "/api/threads/:threadId",
  paramsSchema: updateThreadParamsSchema,
  bodySchema: updateThreadBodySchema,
  responseSchema: threadSummarySchema,
};

export const deleteThreadRoute = {
  method: "DELETE" as const,
  url: "/api/threads/:threadId",
  paramsSchema: updateThreadParamsSchema,
};

export type GetThreadMessagesParams = z.infer<
  typeof getThreadMessagesParamsSchema
>;
export type UpdateThreadParams = z.infer<typeof updateThreadParamsSchema>;
export type UpdateThreadBody = z.infer<typeof updateThreadBodySchema>;
