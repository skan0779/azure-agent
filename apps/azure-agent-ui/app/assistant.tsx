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
  const apiBaseUrl = useMemo(() => {
    const configuredUrl = process.env.NEXT_PUBLIC_AGENT_WEB_URL?.trim();
    if (configuredUrl) {
      return configuredUrl.replace(/\/$/, "");
    }

    if (process.env.NODE_ENV !== "production") {
      return "http://localhost:3001";
    }

    return undefined;
  }, []);
  const transport = useMemo(() => {
    return new AssistantChatTransport({
      api: apiBaseUrl ? `${apiBaseUrl}/api/chat` : "/__missing-agent-web-url__",
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

  if (!apiBaseUrl) {
    return (
      <div className="flex h-dvh items-center justify-center bg-background px-6">
        <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-6 text-card-foreground shadow-sm">
          <h1 className="text-lg font-semibold">Agent Web URL not configured</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Set <code>NEXT_PUBLIC_AGENT_WEB_URL</code> to the public URL of
            the <code>azure-agent-web</code> service before using this UI.
          </p>
        </div>
      </div>
    );
  }

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPersistence chatId={chatId} runtime={runtime} />
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
};
