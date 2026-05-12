import { randomUUID } from "node:crypto";
import type { UIMessage } from "./thread-history.js";
import { uiMessageSchema } from "./thread-history.js";

const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const normalizeUuid = (value?: string | null): string | undefined => {
  const trimmed = value?.trim();
  if (!trimmed || !UUID_REGEX.test(trimmed)) {
    return undefined;
  }

  return trimmed;
};

export const resolveThreadId = (
  threadId?: string | null,
  fallbackId?: string | null,
): string => {
  return normalizeUuid(threadId) ?? normalizeUuid(fallbackId) ?? randomUUID();
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

export const createAutoThreadTitle = (text: string): string => {  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "New chat";
  }

  const tokens = normalized.split(" ");
  if (tokens.length <= AUTO_TITLE_TOKEN_LIMIT) {
    return normalized;
  }

  return `${tokens.slice(0, AUTO_TITLE_TOKEN_LIMIT).join(" ")}...`;
};

export type AttachedFileRef = {
  fileId: string;
  filename: string;
  mimeType?: string;
  size?: number;
};

const toAttachedFileRef = (value: unknown): AttachedFileRef | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const v = value as Record<string, unknown>;
  const fileId =
    typeof v.file_id === "string"
      ? v.file_id
      : typeof v.fileId === "string"
        ? v.fileId
        : "";
  const filename = typeof v.filename === "string" ? v.filename : "";
  if (!fileId || !filename) {
    return null;
  }
  return {
    fileId,
    filename,
    mimeType:
      typeof v.mime_type === "string"
        ? v.mime_type
        : typeof v.mimeType === "string"
          ? v.mimeType
          : undefined,
    size: typeof v.size === "number" ? v.size : undefined,
  };
};

// Matches /api/files/<file_id>/download produced by the UI attachment adapter.
const FILE_PART_URL_PATTERN = /\/api\/files\/([^/?#]+)\/download(?:[?#].*)?$/;

const extractFileIdFromUrl = (url: string): string | null => {
  const match = FILE_PART_URL_PATTERN.exec(url);
  if (!match) return null;
  try {
    return decodeURIComponent(match[1] ?? "") || null;
  } catch {
    return match[1] ?? null;
  }
};

const filePartToAttachedFile = (
  part: Record<string, unknown>,
): Record<string, unknown> | null => {
  // ai-sdk uses `url`, assistant-ui internal type uses `data`.
  const rawUrl =
    typeof part.url === "string"
      ? part.url
      : typeof part.data === "string"
        ? part.data
        : "";
  if (!rawUrl) return null;

  const fileId = extractFileIdFromUrl(rawUrl);
  if (!fileId) return null;

  const filename = typeof part.filename === "string" ? part.filename : "";
  if (!filename) return null;

  const mimeType =
    typeof part.mediaType === "string"
      ? part.mediaType
      : typeof part.mimeType === "string"
        ? part.mimeType
        : undefined;

  return {
    file_id: fileId,
    filename,
    mime_type: mimeType,
  };
};

export const extractAttachedFilesFromMessage = (
  message: unknown,
): AttachedFileRef[] => {
  if (!message || typeof message !== "object" || !("parts" in message)) {
    return [];
  }
  const parts = (message as { parts?: unknown }).parts;
  if (!Array.isArray(parts)) {
    return [];
  }

  const refs = new Map<string, AttachedFileRef>();
  for (const part of parts) {
    if (!part || typeof part !== "object" || !("type" in part)) {
      continue;
    }
    const p = part as Record<string, unknown>;
    const partType = typeof p.type === "string" ? p.type : "";
    let payload: unknown = null;

    if (partType === "data-agent-file") {
      payload = p.data;
    } else if (
      partType === "data" &&
      typeof p.name === "string" &&
      p.name === "agent-file"
    ) {
      payload = p.data;
    } else if (partType === "file") {
      payload = filePartToAttachedFile(p);
    }

    const ref = toAttachedFileRef(payload);
    if (ref && !refs.has(ref.fileId)) {
      refs.set(ref.fileId, ref);
    }
  }

  return [...refs.values()];
};

export const extractLastUserAttachments = (
  messages: unknown[],
): AttachedFileRef[] => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") {
      continue;
    }
    const role =
      "role" in message && typeof message.role === "string"
        ? message.role
        : "";
    if (role !== "user") {
      continue;
    }
    const refs = extractAttachedFilesFromMessage(message);
    if (refs.length > 0) {
      return refs;
    }
    return [];
  }
  return [];
};

export const composeUserQueryWithAttachments = (
  text: string,
  attachments: AttachedFileRef[],
): string => {
  if (attachments.length === 0) {
    return text;
  }
  const list = attachments.map((file) => `- ${file.filename}`).join("\n");
  const baseText = text.trim();
  return baseText
    ? `${baseText}\n\n[Attached files]\n${list}`
    : `[Attached files]\n${list}`;
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

type CitationDataPart = {
  type: "data-citation";
  id: string;
  data: {
    citationId: string;
    href: string;
    title: string;
    snippet?: string;
    domain?: string;
    favicon?: string;
    citationType?: "webpage" | "document" | "article" | "api" | "code" | "other";
  };
};

type CitationRecord = CitationDataPart["data"];

type ToolSnapshot = {
  toolName: string;
  toolCallId: string;
  input?: unknown;
  output?: unknown;
  errorText?: string;
  title?: string;
  providerExecuted?: boolean;
  citations?: CitationRecord[];
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

const getHostname = (value: string): string | undefined => {
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return undefined;
  }
};

const getFaviconUrl = (hostname: string | undefined): string | undefined => {
  if (!hostname) {
    return undefined;
  }

  return `https://www.google.com/s2/favicons?domain=${hostname}&sz=32`;
};

const getResultsFromToolOutput = (output: unknown): unknown[] => {
  if (!output || typeof output !== "object") {
    return [];
  }

  if ("results" in output && Array.isArray(output.results)) {
    return output.results;
  }

  return [];
};

const getContentPartsFromPayload = (payload: Record<string, unknown>): unknown[] => {
  if ("content" in payload && Array.isArray(payload.content)) {
    return payload.content;
  }

  return [];
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

const createCitationRecord = ({
  href,
  title,
  snippet,
}: {
  href: string;
  title: string;
  snippet?: string;
}): CitationRecord => {
  const domain = getHostname(href);

  return {
    citationId: "",
    href,
    title,
    snippet,
    domain,
    favicon: getFaviconUrl(domain),
    citationType: "article",
  };
};

const getCitationRecordKey = (citation: CitationRecord): string =>
  `${citation.href}::${citation.title}`;

const dedupeCitationRecords = (citations: CitationRecord[]): CitationRecord[] => {
  const deduped = new Map<string, CitationRecord>();

  for (const citation of citations) {
    deduped.set(getCitationRecordKey(citation), citation);
  }

  return Array.from(deduped.values());
};

const areCitationRecordListsEqual = (
  left: CitationRecord[] = [],
  right: CitationRecord[] = [],
): boolean => {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((citation, index) => {
    const other = right[index];
    return (
      other &&
      citation.href === other.href &&
      citation.title === other.title &&
      citation.snippet === other.snippet
    );
  });
};

const getCitationRecordsFromAnnotations = (annotations: unknown[]): CitationRecord[] => {
  const citations: CitationRecord[] = [];

  for (const annotation of annotations) {
    if (!annotation || typeof annotation !== "object") {
      continue;
    }

    const href =
      "url" in annotation && typeof annotation.url === "string"
        ? annotation.url
        : "";
    const title =
      "title" in annotation && typeof annotation.title === "string"
        ? annotation.title
        : "";

    if (!href || !title) {
      continue;
    }

    citations.push(
      createCitationRecord({
        href,
        title,
      }),
    );
  }

  return dedupeCitationRecords(citations);
};

const buildWebSearchOutput = (
  input: unknown,
  citations: CitationRecord[],
): Record<string, unknown> => {
  const query =
    input && typeof input === "object" && "query" in input && typeof input.query === "string"
      ? input.query
      : undefined;
  const queries =
    input && typeof input === "object" && "queries" in input && Array.isArray(input.queries)
      ? input.queries
      : undefined;

  return {
    query,
    queries,
    sources: citations.map((citation) => ({
      url: citation.href,
      title: citation.title,
      snippet: citation.snippet,
      domain: citation.domain,
    })),
  };
};

const findLatestProviderToolSnapshot = (
  toolSnapshots: Map<string, ToolSnapshot>,
  toolName: string,
): ToolSnapshot | undefined => {
  const snapshots = Array.from(toolSnapshots.values());

  for (let index = snapshots.length - 1; index >= 0; index -= 1) {
    const snapshot = snapshots[index];
    if (snapshot.toolName === toolName) {
      return snapshot;
    }
  }

  return undefined;
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

const collectToolSnapshotsFromMessages = (
  messages: unknown[],
  toolSnapshots: Map<string, ToolSnapshot>,
): ToolStreamChunk[] => {
  const chunks: ToolStreamChunk[] = [];

  for (const message of messages) {
    if (!message || typeof message !== "object") {
      continue;
    }

    const envelope = message as Record<string, unknown>;
    const messageType =
      typeof envelope.type === "string" ? envelope.type : "";
    const payload =
      "data" in envelope && envelope.data && typeof envelope.data === "object"
        ? (envelope.data as Record<string, unknown>)
        : envelope;

    if (messageType === "AIMessageChunk" || messageType === "ai") {
      const contentParts = getContentPartsFromPayload(payload);

      for (const part of contentParts) {
        if (!part || typeof part !== "object") {
          continue;
        }

        if (!("type" in part) || part.type !== "web_search_call") {
          continue;
        }

        const toolCallId =
          "id" in part && typeof part.id === "string" ? part.id : undefined;
        if (!toolCallId) {
          continue;
        }

        const action =
          "action" in part && part.action && typeof part.action === "object"
            ? (part.action as Record<string, unknown>)
            : {};
        const input = {
          query:
            typeof action.query === "string"
              ? action.query
              : undefined,
          queries: Array.isArray(action.queries) ? action.queries : undefined,
          type: typeof action.type === "string" ? action.type : undefined,
        };
        const previous = toolSnapshots.get(toolCallId);

        toolSnapshots.set(toolCallId, {
          toolCallId,
          toolName: "web_search",
          title: "Web search",
          providerExecuted: true,
          input,
          output: previous?.output,
          errorText: previous?.errorText,
          citations: previous?.citations,
          state:
            previous?.state === "output-available" ||
            previous?.state === "output-error"
              ? previous.state
              : "input-available",
        });

        if (!previous) {
          chunks.push({
            type: "tool-input-available",
            toolCallId,
            toolName: "web_search",
            input,
            dynamic: true,
          });
        }
      }

      const annotationCitations = dedupeCitationRecords(
        contentParts.flatMap((part) => {
          if (!part || typeof part !== "object") {
            return [];
          }

          if (!("type" in part) || part.type !== "text") {
            return [];
          }

          const annotations =
            "annotations" in part && Array.isArray(part.annotations)
              ? part.annotations
              : [];

          return getCitationRecordsFromAnnotations(annotations);
        }),
      );

      if (annotationCitations.length > 0) {
        const snapshot = findLatestProviderToolSnapshot(toolSnapshots, "web_search");
        if (snapshot) {
          const mergedCitations = dedupeCitationRecords([
            ...(snapshot.citations ?? []),
            ...annotationCitations,
          ]);

          if (
            snapshot.state !== "output-available" ||
            !areCitationRecordListsEqual(snapshot.citations, mergedCitations)
          ) {
            snapshot.citations = mergedCitations;
            snapshot.output = buildWebSearchOutput(snapshot.input, mergedCitations);
            snapshot.state = "output-available";
            snapshot.providerExecuted = true;
            snapshot.title = "Web search";
            toolSnapshots.set(snapshot.toolCallId, snapshot);

            chunks.push({
              type: "tool-output-available",
              toolCallId: snapshot.toolCallId,
              output: snapshot.output,
              dynamic: true,
            });
          }
        }
      }
    }

    if (messageType === "ai") {
      const toolCalls =
        "tool_calls" in payload && Array.isArray(payload.tool_calls)
          ? payload.tool_calls
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

        const input = "args" in toolCall ? toolCall.args : undefined;

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
        "tool_call_id" in payload && typeof payload.tool_call_id === "string"
          ? payload.tool_call_id
          : undefined;
      if (!toolCallId) {
        continue;
      }

      const previous = toolSnapshots.get(toolCallId);
      const toolName =
        ("name" in payload && typeof payload.name === "string"
          ? payload.name
          : undefined) ?? previous?.toolName;

      if (!toolName) {
        continue;
      }

      const status =
        "status" in payload && typeof payload.status === "string"
          ? payload.status
          : "success";
      const normalizedContent = normalizeToolContent(
        "content" in payload ? payload.content : undefined,
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

  return chunks;
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
    chunks.push(...collectToolSnapshotsFromMessages(messages, toolSnapshots));
  }

  return chunks;
};

export const collectToolSnapshotsFromMessagesEvent = (
  data: unknown,
  toolSnapshots: Map<string, ToolSnapshot>,
): ToolStreamChunk[] => {
  return Array.isArray(data)
    ? collectToolSnapshotsFromMessages(data, toolSnapshots)
    : [];
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
        title: snapshot.title,
        providerExecuted: snapshot.providerExecuted,
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
        title: snapshot.title,
        providerExecuted: snapshot.providerExecuted,
        state: "output-error",
        input: snapshot.input,
        errorText: snapshot.errorText ?? "Tool execution failed",
      };
    }

    return {
      type: "dynamic-tool",
      toolName: snapshot.toolName,
      toolCallId: snapshot.toolCallId,
      title: snapshot.title,
      providerExecuted: snapshot.providerExecuted,
      state: "input-available",
      input: snapshot.input,
    };
  });
};

export const buildCitationParts = (
  toolSnapshots: Map<string, ToolSnapshot>,
): CitationDataPart[] => {
  const citations: CitationDataPart[] = [];

  for (const snapshot of toolSnapshots.values()) {
    if (snapshot.state !== "output-available") {
      continue;
    }

    const records =
      snapshot.citations && snapshot.citations.length > 0
        ? snapshot.citations
        : getResultsFromToolOutput(snapshot.output)
            .map((result) => {
              if (!result || typeof result !== "object") {
                return undefined;
              }

              const href =
                "url" in result && typeof result.url === "string" ? result.url : "";
              const title =
                "title" in result && typeof result.title === "string"
                  ? result.title
                  : "";

              if (!href || !title) {
                return undefined;
              }

              const snippet =
                "content" in result && typeof result.content === "string"
                  ? result.content
                  : "raw_content" in result && typeof result.raw_content === "string"
                    ? result.raw_content
                    : undefined;

              return createCitationRecord({
                href,
                title,
                snippet,
              });
            })
            .filter((citation): citation is CitationRecord => Boolean(citation));

    for (let index = 0; index < records.length; index += 1) {
      const record = records[index];
      citations.push({
        type: "data-citation",
        id: `${snapshot.toolCallId}:${index}`,
        data: {
          citationId: `${snapshot.toolCallId}:${index}`,
          href: record.href,
          title: record.title,
          snippet: record.snippet,
          domain: record.domain,
          favicon: record.favicon,
          citationType: record.citationType,
        },
      });
    }
  }

  return citations;
};
