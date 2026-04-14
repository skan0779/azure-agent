"use client";

import { useChat } from "@ai-sdk/react";
import {
  AssistantRuntimeProvider,
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
import { LocalThreadListAdapter } from "@/lib/thread-list-adapter.local";
import { ensureLocalActiveThread, localThreadStore } from "@/lib/thread-store.local";

const DEFAULT_USER_ID = "1015520";

const useLocalChatThreadRuntime = ({
  apiBaseUrl,
  userId,
}: {
  apiBaseUrl: string;
  userId: string;
}) => {
  const threadId = useAuiState((s) => s.threadListItem.id);

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
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
  });

  const runtime = useAISDKRuntime(chat);

  if (transport instanceof AssistantChatTransport) {
    transport.setRuntime(runtime);
  }

  return runtime;
};

const ThreadStoreSync = () => {
  const mainThreadId = useAuiState((s) => s.threads.mainThreadId);

  useEffect(() => {
    if (!mainThreadId) {
      return;
    }

    const state = localThreadStore.getState() as Awaited<
      ReturnType<typeof localThreadStore.getState>
    >;
    const existing = state.threads.find((thread) => thread.id === mainThreadId);

    if (existing) {
      localThreadStore.setActiveThread(mainThreadId);
      return;
    }

    localThreadStore.createThread({ id: mainThreadId });
  }, [mainThreadId]);

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
