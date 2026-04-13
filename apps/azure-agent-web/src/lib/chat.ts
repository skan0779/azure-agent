import { randomUUID } from "node:crypto";

export const resolveThreadId = (
  ...candidates: Array<string | undefined | null>
): string => {
  for (const candidate of candidates) {
    const value = candidate?.trim();
    if (!value) {
      continue;
    }

    return value;
  }

  return randomUUID();
};

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

export const extractLangChainChunkText = (message: unknown): string => {
  if (!message || typeof message !== "object") {
    return "";
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

          if ("text" in part && typeof part.text === "string") {
            return part.text;
          }

          return "";
        })
        .join("") || ""
    );
  }

  return "";
};
