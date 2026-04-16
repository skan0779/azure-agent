"use client";

import { useChat } from "@ai-sdk/react";
import {
  AssistantRuntimeProvider,
  useAui,
  useAuiState,
  useAssistantRuntime,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import {
  AssistantChatTransport,
  useAISDKRuntime,
} from "@assistant-ui/react-ai-sdk";
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";

import { Assistant } from "@/app/assistant";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  getThreadMessages,
  updateThread as updateRemoteThread,
} from "@/lib/thread-api";
import { ACTIVE_THREAD_ID_STORAGE_KEY } from "@/lib/thread-store.keys";
import { RemoteApiThreadListAdapter } from "@/lib/thread-list-adapter.remote";
import { localThreadStore } from "@/lib/thread-store.local";

const DEFAULT_USER_ID = "1015520";

const getStoredActiveThreadId = () => {
  if (typeof window === "undefined") {
    return undefined;
  }

  const stored = window.localStorage.getItem(ACTIVE_THREAD_ID_STORAGE_KEY);
  return stored?.trim() || undefined;
};

const extractJobId = (value: unknown): string | null => {
  if (!value || typeof value !== "object") {
    return null;
  }

  if (!("jobId" in value) || typeof value.jobId !== "string") {
    return null;
  }

  return value.jobId;
};

const cancelChatJob = async ({
  apiBaseUrl,
  jobId,
  userId,
}: {
  apiBaseUrl: string;
  jobId: string;
  userId: string;
}) => {
  try {
    const response = await fetch(`${apiBaseUrl}/api/chat/cancel`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        jobId,
        userId,
      }),
    });

    if (!response.ok) {
      console.warn("Failed to cancel chat job", {
        jobId,
        status: response.status,
      });
    }
  } catch (error) {
    console.warn("Failed to cancel chat job", {
      jobId,
      error,
    });
  }
};

const useLocalChatThreadRuntime = ({
  apiBaseUrl,
  userId,
}: {
  apiBaseUrl: string;
  userId: string;
}) => {
  const assistantRuntime = useAssistantRuntime();
  const threadId = useAuiState((s) => s.threadListItem.id);
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const runtimeThreadKey = remoteId ?? threadId;
  const currentJobIdRef = useRef<string | null>(null);

  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: `${apiBaseUrl}/api/chat`,
        body: {
          userId,
          threadId: remoteId,
        },
      }),
    [apiBaseUrl, remoteId, userId],
  );

  const chat = useChat({
    id: threadId,
    messages: [],
    transport,
    onData: (dataPart) => {
      if (dataPart.type !== "data-metadata") {
        return;
      }

      const jobId = extractJobId(dataPart.data);
      if (jobId) {
        currentJobIdRef.current = jobId;
      }
    },
    onFinish: () => {
      currentJobIdRef.current = null;
    },
    onError: () => {
      currentJobIdRef.current = null;
    },
  });

  const stop = useCallback(async () => {
    const jobId = currentJobIdRef.current;
    currentJobIdRef.current = null;

    await Promise.allSettled([
      chat.stop(),
      jobId
        ? cancelChatJob({
            apiBaseUrl,
            jobId,
            userId,
          })
        : Promise.resolve(),
    ]);
  }, [apiBaseUrl, chat, userId]);

  const lastObservedUserMessageIdRef = useRef<string | null>(null);
  const lastUpdatedUserMessageKeyRef = useRef<string | null>(null);
  const lastRuntimeThreadKeyRef = useRef<string | null>(null);
  const setMessagesRef = useRef(chat.setMessages);

  useEffect(() => {
    setMessagesRef.current = chat.setMessages;
  }, [chat.setMessages]);

  const syncMessagesFromServer = useCallback(
    async (
      targetThreadId: string,
      { context }: { context: string },
    ) => {
      try {
        const messages = await getThreadMessages({
          apiBaseUrl,
          userId,
          threadId: targetThreadId,
        });

        setMessagesRef.current(messages);
      } catch (error) {
        console.warn(`Failed to ${context}`, {
          threadId: targetThreadId,
          error,
        });
      }
    },
    [apiBaseUrl, userId],
  );

  const latestUserMessageId =
    [...chat.messages]
      .reverse()
      .find((message) => message.role === "user")?.id ?? null;

  useEffect(() => {
    if (remoteId) {
      return;
    }

    void assistantRuntime.threads.mainItem.initialize().catch((error) => {
      console.warn("Failed to initialize remote thread", {
        threadId,
        error,
      });
    });
  }, [assistantRuntime, remoteId, threadId]);

  useEffect(() => {
    if (!remoteId) {
      return;
    }

    if (chat.messages.length > 0) {
      return;
    }

    let cancelled = false;

    void syncMessagesFromServer(remoteId, {
      context: "load thread messages",
    }).then(() => {
      if (cancelled) {
        return;
      }
    });

    return () => {
      cancelled = true;
    };
  }, [chat.messages.length, remoteId, syncMessagesFromServer]);

  useEffect(() => {
    currentJobIdRef.current = null;
  }, [runtimeThreadKey]);

  useEffect(() => {
    if (lastRuntimeThreadKeyRef.current !== runtimeThreadKey) {
      lastRuntimeThreadKeyRef.current = runtimeThreadKey;
      lastObservedUserMessageIdRef.current = latestUserMessageId;
      lastUpdatedUserMessageKeyRef.current =
        remoteId && latestUserMessageId
          ? `${remoteId}:${latestUserMessageId}`
          : null;
      return;
    }

    if (!latestUserMessageId) {
      return;
    }

    const userMessageChanged =
      lastObservedUserMessageIdRef.current !== latestUserMessageId;
    if (userMessageChanged) {
      lastObservedUserMessageIdRef.current = latestUserMessageId;
    }

    if (!remoteId) {
      return;
    }

    const updateKey = `${remoteId}:${latestUserMessageId}`;
    if (lastUpdatedUserMessageKeyRef.current === updateKey) {
      return;
    }

    if (!userMessageChanged && lastUpdatedUserMessageKeyRef.current) {
      return;
    }

    const updatedAt = new Date().toISOString();
    void updateRemoteThread({
      apiBaseUrl,
      userId,
      threadId: remoteId,
      updatedAt,
    }).catch((error) => {
      console.warn("Failed to update thread timestamp", {
        threadId: remoteId,
        error,
      });
    });
    lastUpdatedUserMessageKeyRef.current = updateKey;
  }, [apiBaseUrl, latestUserMessageId, remoteId, runtimeThreadKey, userId]);

  const runtime = useAISDKRuntime({
    ...chat,
    stop,
  });

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

    localThreadStore.setActiveThread(remoteId);
  }, [remoteId]);

  return null;
};

const InitialThreadSwitch = ({
  storedThreadId,
}: {
  storedThreadId?: string;
}) => {
  const aui = useAui();
  const isLoading = useAuiState((s) => s.threads.isLoading);
  const mainThreadId = useAuiState((s) => s.threads.mainThreadId);
  const hasSwitchedRef = useRef(false);

  useEffect(() => {
    if (hasSwitchedRef.current || isLoading || !storedThreadId) {
      return;
    }

    if (mainThreadId === storedThreadId) {
      hasSwitchedRef.current = true;
      return;
    }

    hasSwitchedRef.current = true;
    void aui.threads().switchToThread(storedThreadId);
  }, [aui, isLoading, mainThreadId, storedThreadId]);

  return null;
};

const ChatShellRuntime = ({
  apiBaseUrl,
  storedThreadId,
}: {
  apiBaseUrl: string;
  storedThreadId?: string;
}) => {
  const adapter = useMemo(
    () =>
      new RemoteApiThreadListAdapter({
        apiBaseUrl,
        userId: DEFAULT_USER_ID,
      }),
    [apiBaseUrl],
  );

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: function RuntimeHook() {
      return useLocalChatThreadRuntime({
        apiBaseUrl,
        userId: DEFAULT_USER_ID,
      });
    },
    adapter,
    allowNesting: true,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <InitialThreadSwitch storedThreadId={storedThreadId} />
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
  const isHydrated = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
  const storedThreadId = useMemo(() => {
    if (!isHydrated) {
      return undefined;
    }

    return getStoredActiveThreadId();
  }, [isHydrated]);

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

  if (!isHydrated) {
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
      storedThreadId={storedThreadId}
    />
  );
};
