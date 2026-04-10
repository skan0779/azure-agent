"use client";

import { useEffect, useMemo, useRef } from "react";
import { AssistantRuntimeProvider, useAuiState } from "@assistant-ui/react";
import {
  useChatRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { type UIMessage, lastAssistantMessageIsCompleteWithToolCalls } from "ai";
import { Thread } from "@/components/assistant-ui/thread";
import { getThreadStorageKey } from "@/lib/chat-storage";

type AssistantProps = {
  chatId: string;
  userId: string;
  onSessionTouched: (chatId: string, titleHint?: string) => void;
};

const extractLastUserText = (messages: UIMessage[]) => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "user") {
      continue;
    }

    if (Array.isArray(message.parts)) {
      const text = message.parts
        .map((part) => {
          if (!part || typeof part !== "object") {
            return "";
          }

          if ("type" in part && part.type === "text" && "text" in part) {
            return typeof part.text === "string" ? part.text : "";
          }

          return "";
        })
        .join("")
        .trim();

      if (text) {
        return text;
      }
    }
  }

  return undefined;
};

const ThreadPersistence = ({
  chatId,
  runtime,
}: {
  chatId: string;
  runtime: ReturnType<typeof useChatRuntime>;
}) => {
  const messageSnapshot = useAuiState((state) =>
    state.thread.messages.map((message) => ({
      id: message.id,
      role: message.role,
      status: message.status,
      parts: message.parts.map((part) => {
        if (part.type === "text") {
          return `${part.type}:${part.text}`;
        }

        return part.type;
      }),
    })),
  );
  const didRestoreRef = useRef(false);

  useEffect(() => {
    const storageKey = getThreadStorageKey(chatId);
    const raw = window.localStorage.getItem(storageKey);
    if (raw) {
      try {
        runtime.thread.import(JSON.parse(raw));
      } catch {
        window.localStorage.removeItem(storageKey);
      }
    }

    didRestoreRef.current = true;
  }, [chatId, runtime]);

  useEffect(() => {
    if (!didRestoreRef.current) {
      return;
    }

    const storageKey = getThreadStorageKey(chatId);
    window.localStorage.setItem(
      storageKey,
      JSON.stringify(runtime.thread.export()),
    );
  }, [chatId, messageSnapshot, runtime]);

  return null;
};

export const Assistant = ({
  chatId,
  userId,
  onSessionTouched,
}: AssistantProps) => {
  const apiBaseUrl = process.env.NEXT_PUBLIC_AGENT_WEB_URL?.replace(/\/$/, "");
  const transport = useMemo(() => {
    return new AssistantChatTransport({
      api: apiBaseUrl ? `${apiBaseUrl}/api/chat` : "/api/chat",
      body: {
        userId,
      },
      prepareSendMessagesRequest: ({ body, id, messages }) => {
        const titleHint = extractLastUserText(messages);
        onSessionTouched(id, titleHint);

        return {
          body: {
            ...body,
            userId,
          },
        };
      },
    });
  }, [apiBaseUrl, onSessionTouched, userId]);

  const runtime = useChatRuntime({
    id: chatId,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
    transport,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPersistence chatId={chatId} runtime={runtime} />
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};
