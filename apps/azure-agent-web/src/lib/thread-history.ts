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

const filePartSchema = z.object({
  type: z.literal("file"),
  mimeType: z.string().min(1),
  filename: z.string().optional(),
  data: z.string().min(1),
});

const stepStartPartSchema = z.object({
  type: z.literal("step-start"),
});

const dataPartSchema = z.union([
  z
    .object({
      type: z.string().regex(/^data-/),
      id: z.string().optional(),
      data: z.unknown(),
    })
    .passthrough(),
  z
    .object({
      type: z.literal("data"),
      name: z.string().min(1),
      id: z.string().optional(),
      data: z.unknown(),
    })
    .passthrough(),
]);

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

const normalizeLegacyCitationPart = (part: Record<string, unknown>) => {
  if (part.type === "data-citation") {
    return part;
  }

  if (part.type === "data" && part.name === "citation") {
    return {
      type: "data-citation",
      id: typeof part.id === "string" ? part.id : undefined,
      data: part.data,
    };
  }

  if (part.type === "citation") {
    const citationId =
      typeof part.citationId === "string" && part.citationId.length > 0
        ? part.citationId
        : typeof part.id === "string" && part.id.length > 0
          ? part.id
          : undefined;

    return {
      type: "data-citation",
      id: citationId,
      data: {
        citationId,
        href: typeof part.href === "string" ? part.href : undefined,
        title: typeof part.title === "string" ? part.title : undefined,
        snippet: typeof part.snippet === "string" ? part.snippet : undefined,
        domain: typeof part.domain === "string" ? part.domain : undefined,
        favicon: typeof part.favicon === "string" ? part.favicon : undefined,
        author: typeof part.author === "string" ? part.author : undefined,
        publishedAt:
          typeof part.publishedAt === "string" ? part.publishedAt : undefined,
        citationType:
          typeof part.citationType === "string" ? part.citationType : undefined,
      },
    };
  }

  return part;
};

export const normalizeStoredUiMessage = (message: {
  id: string;
  role: "system" | "user" | "assistant";
  metadata?: unknown;
  parts: unknown;
}) => {
  const normalizedParts = Array.isArray(message.parts)
    ? message.parts.map((part) => {
        if (!part || typeof part !== "object") {
          return part;
        }

        return normalizeLegacyCitationPart(part as Record<string, unknown>);
      })
    : message.parts;

  return {
    ...message,
    parts: normalizedParts,
  };
};

export const listThreadMessagesResponseSchema = z.array(uiMessageSchema);

export type ThreadSummary = z.infer<typeof threadSummarySchema>;
export type ListThreadsResponse = z.infer<typeof listThreadsResponseSchema>;
export type UIMessagePart = z.infer<typeof uiMessagePartSchema>;
export type UIMessage = z.infer<typeof uiMessageSchema>;
export type ListThreadMessagesResponse = z.infer<
  typeof listThreadMessagesResponseSchema
>;
