import { z } from "zod";

// Thread/history read contracts for future server-backed storage.
// These shapes intentionally track the AI SDK UIMessage model because the
// current UI already persists and restores `UIMessage[]` in localStorage.

export const threadIdSchema = z.string().uuid();
export const isoDateTimeSchema = z.string().datetime({ offset: true });

export const threadTitleSourceSchema = z.enum([
  "manual",
  "first-user-message",
  "generated",
]);

export const threadSummarySchema = z.object({
  id: threadIdSchema,
  title: z.string().min(1),
  createdAt: isoDateTimeSchema,
  updatedAt: isoDateTimeSchema,
  lastJobId: z.string().min(1).optional(),
  titleSource: threadTitleSourceSchema.optional(),
});

export const listThreadsResponseSchema = z.array(threadSummarySchema);

const textPartSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
  state: z.enum(["streaming", "done"]).optional(),
});

const reasoningPartSchema = z.object({
  type: z.literal("reasoning"),
  text: z.string(),
  state: z.enum(["streaming", "done"]).optional(),
});

const citationTypeSchema = z.enum([
  "webpage",
  "document",
  "article",
  "api",
  "code",
  "other",
]);

const citationPartSchema = z.object({
  type: z.literal("citation"),
  citationId: z.string().min(1),
  href: z.string().url(),
  title: z.string().min(1),
  snippet: z.string().optional(),
  domain: z.string().optional(),
  favicon: z.string().optional(),
  author: z.string().optional(),
  publishedAt: z.string().optional(),
  citationType: citationTypeSchema.optional(),
});

const filePartSchema = z.object({
  type: z.literal("file"),
  mimeType: z.string().min(1),
  filename: z.string().optional(),
  data: z.string().min(1),
});

const stepStartPartSchema = z.object({
  type: z.literal("step-start"),
});

const dataPartSchema = z
  .object({
    type: z.string().regex(/^data-/),
    id: z.string().optional(),
    data: z.unknown(),
  })
  .passthrough();

const toolApprovalPendingSchema = z.object({
  id: z.string().min(1),
  approved: z.undefined().optional(),
  reason: z.undefined().optional(),
});

const toolApprovalGrantedSchema = z.object({
  id: z.string().min(1),
  approved: z.literal(true),
  reason: z.string().optional(),
});

const toolApprovalDeniedSchema = z.object({
  id: z.string().min(1),
  approved: z.literal(false),
  reason: z.string().optional(),
});

const dynamicToolBaseSchema = z.object({
  type: z.literal("dynamic-tool"),
  toolName: z.string().min(1),
  toolCallId: z.string().min(1),
  title: z.string().optional(),
  providerExecuted: z.boolean().optional(),
});

const dynamicToolPartSchema = z.discriminatedUnion("state", [
  dynamicToolBaseSchema.extend({
    state: z.literal("input-streaming"),
    input: z.unknown().optional(),
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("input-available"),
    input: z.unknown(),
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("approval-requested"),
    input: z.unknown(),
    approval: toolApprovalPendingSchema,
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("approval-responded"),
    input: z.unknown(),
    approval: z.union([toolApprovalGrantedSchema, toolApprovalDeniedSchema]),
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("output-available"),
    input: z.unknown(),
    output: z.unknown(),
    preliminary: z.boolean().optional(),
    approval: toolApprovalGrantedSchema.optional(),
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("output-error"),
    input: z.unknown(),
    errorText: z.string().min(1),
    approval: toolApprovalGrantedSchema.optional(),
  }),
  dynamicToolBaseSchema.extend({
    state: z.literal("output-denied"),
    input: z.unknown(),
    approval: toolApprovalDeniedSchema,
  }),
]);

export const uiMessagePartSchema = z.union([
  textPartSchema,
  reasoningPartSchema,
  dynamicToolPartSchema,
  citationPartSchema,
  filePartSchema,
  dataPartSchema,
  stepStartPartSchema,
]);

export const uiMessageSchema = z.object({
  id: z.string().min(1),
  role: z.enum(["system", "user", "assistant"]),
  metadata: z.unknown().optional(),
  parts: z.array(uiMessagePartSchema),
});

export const listThreadMessagesResponseSchema = z.array(uiMessageSchema);

export type ThreadSummary = z.infer<typeof threadSummarySchema>;
export type ListThreadsResponse = z.infer<typeof listThreadsResponseSchema>;
export type UIMessagePart = z.infer<typeof uiMessagePartSchema>;
export type UIMessage = z.infer<typeof uiMessageSchema>;
export type ListThreadMessagesResponse = z.infer<
  typeof listThreadMessagesResponseSchema
>;
