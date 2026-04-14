"use client";

import { useChat } from "@ai-sdk/react";
import {
  AssistantRuntimeProvider,
  useAui,
  useAuiState,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import {
  AssistantChatTransport,
  useAISDKRuntime,
} from "@assistant-ui/react-ai-sdk";
import { lastAssistantMessageIsCompleteWithToolCalls } from "ai";
import { useEffect, useMemo, useState } from "react";

import { Assistant } from "@/app/assistant";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import { localThreadMessageStore } from "@/lib/thread-message-store.local";
import { LocalThreadListAdapter } from "@/lib/thread-list-adapter.local";
import {
  DEFAULT_THREAD_TITLE,
  ensureLocalActiveThread,
  localThreadStore,
} from "@/lib/thread-store.local";

const DEFAULT_USER_ID = "1015520";
const AUTO_TITLE_TOKEN_LIMIT = 7;

const getLocalThreadStoreState = () => {
  return localThreadStore.getState() as Awaited<
    ReturnType<typeof localThreadStore.getState>
  >;
};

const createAutoThreadTitle = (text: string) => {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return DEFAULT_THREAD_TITLE;
  }

  const tokens = normalized.split(" ");
  if (tokens.length <= AUTO_TITLE_TOKEN_LIMIT) {
    return normalized;
  }

  return `${tokens.slice(0, AUTO_TITLE_TOKEN_LIMIT).join(" ")}...`;
};

const getFirstUserMessageTitle = (
  messages: readonly {
    role: string;
    content: readonly { type: string; text?: string }[];
  }[],
) => {
  const firstUserMessage = messages.find((message) => message.role === "user");
  if (!firstUserMessage) {
    return null;
  }

  const text = firstUserMessage.content
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => part.text?.trim() ?? "")
    .filter(Boolean)
    .join(" ")
    .trim();

  if (!text) {
    return null;
  }

  return createAutoThreadTitle(text);
};

const useLocalChatThreadRuntime = ({
  apiBaseUrl,
  userId,
}: {
  apiBaseUrl: string;
  userId: string;
}) => {
  const threadId = useAuiState((s) => s.threadListItem.id);
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const storageThreadId = remoteId ?? threadId;
  const initialMessages = useMemo(
    () => localThreadMessageStore.getMessages(storageThreadId),
    [storageThreadId],
  );

  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: `${apiBaseUrl}/api/chat`,
        body: {
          userId,
          threadId,
        },
      }),
    [apiBaseUrl, threadId, userId],
  );

  const chat = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
  });

  useEffect(() => {
    localThreadMessageStore.setMessages(storageThreadId, chat.messages);
  }, [chat.messages, storageThreadId]);

  const runtime = useAISDKRuntime(chat);

  if (transport instanceof AssistantChatTransport) {
    transport.setRuntime(runtime);
  }

  return runtime;
};

const ThreadStoreSync = () => {
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);

  useEffect(() => {
    if (!remoteId) {
      return;
    }

    const state = getLocalThreadStoreState();
    const existing = state.threads.find((thread) => thread.id === remoteId);

    if (existing) {
      localThreadStore.setActiveThread(remoteId);
      return;
    }

    localThreadStore.createThread({ id: remoteId });
  }, [remoteId]);

  return null;
};

const ThreadTitleSync = () => {
  const aui = useAui();
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const firstUserTitle = useAuiState((s) =>
    getFirstUserMessageTitle(
      s.thread.messages.map((message) => ({
        role: message.role,
        content: message.content.map((part) => ({
          type: part.type,
          text: "text" in part ? part.text : undefined,
        })),
      })),
    ),
  );

  useEffect(() => {
    if (!remoteId || !firstUserTitle) {
      return;
    }

    const state = getLocalThreadStoreState();
    const thread = state.threads.find((item) => item.id === remoteId);
    if (!thread) {
      return;
    }

    const hasManualTitle =
      thread.titleSource === "manual" && thread.title !== DEFAULT_THREAD_TITLE;
    if (hasManualTitle) {
      return;
    }

    if (
      thread.titleSource === "first-user-message" &&
      thread.title === firstUserTitle
    ) {
      return;
    }

    if (thread.title === firstUserTitle) {
      localThreadStore.updateThread(remoteId, {
        titleSource: "first-user-message",
      });
      return;
    }

    aui.threads().item("main").rename(firstUserTitle);
    localThreadStore.updateThread(remoteId, {
      title: firstUserTitle,
      titleSource: "first-user-message",
    });
  }, [aui, firstUserTitle, remoteId]);

  return null;
};

const ChatShellRuntime = ({
  apiBaseUrl,
  initialThreadId,
}: {
  apiBaseUrl: string;
  initialThreadId: string;
}) => {
  const adapter = useMemo(() => new LocalThreadListAdapter(), []);

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: function RuntimeHook() {
      return useLocalChatThreadRuntime({
        apiBaseUrl,
        userId: DEFAULT_USER_ID,
      });
    },
    adapter,
    threadId: initialThreadId,
    allowNesting: true,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadStoreSync />
      <ThreadTitleSync />
      <div className="flex h-dvh bg-[#212121] text-[#ececec]">
        <ThreadListSidebar />
        <main className="min-w-0 flex-1">
          <Assistant />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
};

export const ChatShell = () => {
  const [initialThreadId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }

    return ensureLocalActiveThread(localThreadStore).id;
  });

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

  if (!initialThreadId) {
    return null;
  }

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
    <ChatShellRuntime
      apiBaseUrl={apiBaseUrl}
      initialThreadId={initialThreadId}
    />
  );
};
