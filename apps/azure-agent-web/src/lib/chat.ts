import { randomUUID } from "node:crypto";
import type { UIMessage } from "./thread-history.js";
import { uiMessageSchema } from "./thread-history.js";

export const resolveThreadId = (threadId?: string | null): string => {
  const value = threadId?.trim();
  return value || randomUUID();
};

const AUTO_TITLE_TOKEN_LIMIT = 7;

const extractTextFromParts = (parts: unknown[]): string | undefined => {
  const text = parts
    .map((part) => {
      if (!part || typeof part !== "object") {
        return "";
      }

      if (
        "type" in part &&
        part.type === "text" &&
        "text" in part &&
        typeof part.text === "string"
      ) {
        return part.text;
      }

      if ("text" in part && typeof part.text === "string") {
        return part.text;
      }

      return "";
    })
    .join("");

  return text || undefined;
};

export const extractLastUserText = (messages: unknown[]): string | undefined => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") {
      continue;
    }

    const role =
      "role" in message && typeof message.role === "string" ? message.role : "";
    if (role !== "user") {
      continue;
    }

    const content =
      "content" in message && typeof message.content === "string"
        ? message.content
        : undefined;
    if (content) {
      return content;
    }

    const parts =
      "parts" in message && Array.isArray(message.parts) ? message.parts : [];
    const text = extractTextFromParts(parts);
    if (text) {
      return text;
    }
  }

  return undefined;
};

export const createAutoThreadTitle = (text: string): string => {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "New chat";
  }

  const tokens = normalized.split(" ");
  if (tokens.length <= AUTO_TITLE_TOKEN_LIMIT) {
    return normalized;
  }

  return `${tokens.slice(0, AUTO_TITLE_TOKEN_LIMIT).join(" ")}...`;
};

export const extractLastUserMessage = (
  messages: unknown[],
): UIMessage | undefined => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const candidate = messages[index];
    const parsed = uiMessageSchema.safeParse(candidate);
    if (parsed.success && parsed.data.role === "user") {
      return parsed.data;
    }

    if (!candidate || typeof candidate !== "object") {
      continue;
    }

    const role =
      "role" in candidate && typeof candidate.role === "string"
        ? candidate.role
        : "";
    if (role !== "user") {
      continue;
    }

    const content =
      "content" in candidate && typeof candidate.content === "string"
        ? candidate.content
        : undefined;
    if (content) {
      return {
        id:
          "id" in candidate && typeof candidate.id === "string"
            ? candidate.id
            : randomUUID(),
        role: "user",
        parts: [
          {
            type: "text",
            text: content,
          },
        ],
      };
    }

    const parts =
      "parts" in candidate && Array.isArray(candidate.parts)
        ? candidate.parts
        : [];
    const text = extractTextFromParts(parts);
    if (!text) {
      continue;
    }

    return {
      id:
        "id" in candidate && typeof candidate.id === "string"
          ? candidate.id
          : randomUUID(),
      role: "user",
      parts: [
        {
          type: "text",
          text,
        },
      ],
    };
  }

  return undefined;
};

export const extractLangChainChunkText = (
  message: unknown,
  metadata?: unknown,
): string => {
  if (!message || typeof message !== "object") {
    return "";
  }

  const messageType =
    "type" in message && typeof message.type === "string" ? message.type : "";
  if (messageType !== "AIMessageChunk") {
    return "";
  }

  if (metadata && typeof metadata === "object") {
    const langgraphNode =
      "langgraph_node" in metadata && typeof metadata.langgraph_node === "string"
        ? metadata.langgraph_node
        : undefined;

    if (langgraphNode && langgraphNode !== "model") {
      return "";
    }
  }

  const payload =
    "data" in message && message.data && typeof message.data === "object"
      ? message.data
      : message;

  if (
    "content" in payload &&
    typeof payload.content === "string" &&
    payload.content
  ) {
    return payload.content;
  }

  if ("content" in payload && Array.isArray(payload.content)) {
    return (
      payload.content
        .map((part) => {
          if (typeof part === "string") {
            return part;
          }

          if (!part || typeof part !== "object") {
            return "";
          }

          if (
            "type" in part &&
            part.type === "text" &&
            "text" in part &&
            typeof part.text === "string"
          ) {
            return part.text;
          }

          return "";
        })
        .join("") || ""
    );
  }

  return "";
};

type DynamicToolPart = Extract<
  UIMessage["parts"][number],
  { type: "dynamic-tool" }
>;

type ToolSnapshot = {
  toolName: string;
  toolCallId: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  state:
    | "input-available"
    | "output-available"
    | "output-error";
};

export type ToolStreamChunk =
  | {
      type: "tool-input-available";
      toolCallId: string;
      toolName: string;
      input: unknown;
      dynamic: true;
    }
  | {
      type: "tool-output-available";
      toolCallId: string;
      output: unknown;
      dynamic: true;
    }
  | {
      type: "tool-output-error";
      toolCallId: string;
      errorText: string;
      dynamic: true;
    };

const normalizeToolContent = (content: unknown) => {
  if (typeof content === "string") {
    const trimmed = content.trim();
    if (
      (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
      (trimmed.startsWith("[") && trimmed.endsWith("]"))
    ) {
      try {
        return JSON.parse(trimmed) as unknown;
      } catch {
        return content;
      }
    }

    return content;
  }

  if (Array.isArray(content) || (content && typeof content === "object")) {
    return content;
  }

  return String(content ?? "");
};

const getToolMessagesFromUpdateNode = (value: unknown): unknown[] => {
  if (!value || typeof value !== "object") {
    return [];
  }

  if (
    "messages" in value &&
    Array.isArray(value.messages)
  ) {
    return value.messages;
  }

  return [];
};

export const collectToolSnapshotsFromUpdate = (
  update: unknown,
  toolSnapshots: Map<string, ToolSnapshot>,
): ToolStreamChunk[] => {
  const chunks: ToolStreamChunk[] = [];
  if (!update || typeof update !== "object") {
    return chunks;
  }

  for (const nodeValue of Object.values(update)) {
    const messages = getToolMessagesFromUpdateNode(nodeValue);

    for (const message of messages) {
      if (!message || typeof message !== "object") {
        continue;
      }

      const messageType =
        "type" in message && typeof message.type === "string"
          ? message.type
          : "";

      if (messageType === "ai") {
        const toolCalls =
          "tool_calls" in message && Array.isArray(message.tool_calls)
            ? message.tool_calls
            : [];

        for (const toolCall of toolCalls) {
          if (!toolCall || typeof toolCall !== "object") {
            continue;
          }

          const toolCallId =
            "id" in toolCall && typeof toolCall.id === "string"
              ? toolCall.id
              : undefined;
          const toolName =
            "name" in toolCall && typeof toolCall.name === "string"
              ? toolCall.name
              : undefined;

          if (!toolCallId || !toolName) {
            continue;
          }

          const input =
            "args" in toolCall
              ? toolCall.args
              : undefined;

          toolSnapshots.set(toolCallId, {
            toolCallId,
            toolName,
            input,
            output: toolSnapshots.get(toolCallId)?.output,
            errorText: toolSnapshots.get(toolCallId)?.errorText,
            state:
              toolSnapshots.get(toolCallId)?.state === "output-available" ||
              toolSnapshots.get(toolCallId)?.state === "output-error"
                ? toolSnapshots.get(toolCallId)!.state
                : "input-available",
          });
          chunks.push({
            type: "tool-input-available",
            toolCallId,
            toolName,
            input,
            dynamic: true,
          });
        }

        continue;
      }

      if (messageType === "tool") {
        const toolCallId =
          "tool_call_id" in message && typeof message.tool_call_id === "string"
            ? message.tool_call_id
            : undefined;
        if (!toolCallId) {
          continue;
        }

        const previous = toolSnapshots.get(toolCallId);
        const toolName =
          ("name" in message && typeof message.name === "string"
            ? message.name
            : undefined) ?? previous?.toolName;

        if (!toolName) {
          continue;
        }

        const status =
          "status" in message && typeof message.status === "string"
            ? message.status
            : "success";
        const normalizedContent = normalizeToolContent(
          "content" in message ? message.content : undefined,
        );

        toolSnapshots.set(toolCallId, {
          toolCallId,
          toolName,
          input: previous?.input,
          output: status === "success" ? normalizedContent : undefined,
          errorText:
            status === "success"
              ? undefined
              : typeof normalizedContent === "string"
                ? normalizedContent
                : JSON.stringify(normalizedContent),
          state: status === "success" ? "output-available" : "output-error",
        });

        if (status === "success") {
          chunks.push({
            type: "tool-output-available",
            toolCallId,
            output: normalizedContent,
            dynamic: true,
          });
        } else {
          chunks.push({
            type: "tool-output-error",
            toolCallId,
            errorText:
              typeof normalizedContent === "string"
                ? normalizedContent
                : JSON.stringify(normalizedContent),
            dynamic: true,
          });
        }
      }
    }
  }

  return chunks;
};

export const buildDynamicToolParts = (
  toolSnapshots: Map<string, ToolSnapshot>,
): DynamicToolPart[] => {
  return Array.from(toolSnapshots.values()).map((snapshot) => {
    if (snapshot.state === "output-available") {
      return {
        type: "dynamic-tool",
        toolName: snapshot.toolName,
        toolCallId: snapshot.toolCallId,
        state: "output-available",
        input: snapshot.input,
        output: snapshot.output,
      };
    }

    if (snapshot.state === "output-error") {
      return {
        type: "dynamic-tool",
        toolName: snapshot.toolName,
        toolCallId: snapshot.toolCallId,
        state: "output-error",
        input: snapshot.input,
        errorText: snapshot.errorText ?? "Tool execution failed",
      };
    }

    return {
      type: "dynamic-tool",
      toolName: snapshot.toolName,
      toolCallId: snapshot.toolCallId,
      state: "input-available",
      input: snapshot.input,
    };
  });
};
