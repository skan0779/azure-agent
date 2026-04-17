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
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { Assistant } from "@/app/assistant";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { getThreadMessages } from "@/lib/thread-api";
import { ACTIVE_THREAD_ID_STORAGE_KEY } from "@/lib/thread-store.keys";
import { RemoteApiThreadListAdapter } from "@/lib/thread-list-adapter.remote";

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

const setStoredActiveThreadId = (threadId: string) => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(ACTIVE_THREAD_ID_STORAGE_KEY, threadId);
};

const clearStoredActiveThreadId = () => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.removeItem(ACTIVE_THREAD_ID_STORAGE_KEY);
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
  const chatStoreThreadIdRef = useRef<string | null>(null);
  const chatStoreKeyRef = useRef<string>("");
  const currentJobIdRef = useRef<string | null>(null);
  const remoteIdRef = useRef<string | null>(remoteId ?? null);
  const syncMessagesFromServerRef = useRef<
    | ((
        targetThreadId: string,
        options: { context: string; mode?: "replace" | "merge-latest-assistant-citations" },
      ) => Promise<void>)
    | null
  >(null);
  const messagesRef = useRef<ReturnType<typeof useChat>["messages"]>([]);

  if (chatStoreThreadIdRef.current !== threadId) {
    chatStoreThreadIdRef.current = threadId;
    chatStoreKeyRef.current = remoteId ?? `draft:${threadId}`;
  }

  const chatStoreKey = chatStoreKeyRef.current;

  const transport = useMemo(
    () =>
      new AssistantChatTransport({
        api: `${apiBaseUrl}/api/chat`,
        body: {
          userId,
        },
      }),
    [apiBaseUrl, userId],
  );

  const chat = useChat({
    id: chatStoreKey,
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
      const targetThreadId = remoteIdRef.current;
      if (!targetThreadId || !syncMessagesFromServerRef.current) {
        return;
      }

      void syncMessagesFromServerRef.current(targetThreadId, {
        context: "refresh thread messages after response",
        mode: "merge-latest-assistant-citations",
      });
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

  useEffect(() => {
    messagesRef.current = chat.messages;
  }, [chat.messages]);

  const syncMessagesFromServer = useCallback(
    async (
      targetThreadId: string,
      {
        context,
        mode = "replace",
      }: { context: string; mode?: "replace" | "merge-latest-assistant-citations" },
    ) => {
      try {
        const messages = await getThreadMessages({
          apiBaseUrl,
          userId,
          threadId: targetThreadId,
        });

        if (mode === "replace") {
          setMessagesRef.current(messages);
          return;
        }

        const currentMessages = messagesRef.current;
        const currentAssistantIndex = [...currentMessages]
          .map((message, index) => ({ message, index }))
          .reverse()
          .find(({ message }) => message.role === "assistant")?.index;
        const serverAssistant = [...messages]
          .reverse()
          .find((message) => message.role === "assistant");

        if (
          currentAssistantIndex === undefined ||
          currentAssistantIndex < 0 ||
          !serverAssistant
        ) {
          return;
        }

        const mergedMessages = currentMessages.map((message, index) =>
          index === currentAssistantIndex
            ? {
                ...message,
                parts: [
                  ...message.parts.filter(
                    (part) =>
                      !part ||
                      typeof part !== "object" ||
                      !("type" in part) ||
                      part.type !== "data-citation",
                  ),
                  ...serverAssistant.parts.filter(
                    (part) =>
                      part &&
                      typeof part === "object" &&
                      "type" in part &&
                      part.type === "data-citation",
                  ),
                ],
              }
            : message,
        );

        setMessagesRef.current(mergedMessages);
      } catch (error) {
        console.warn(`Failed to ${context}`, {
          threadId: targetThreadId,
          error,
        });
      }
    },
    [apiBaseUrl, userId],
  );

  useEffect(() => {
    remoteIdRef.current = remoteId ?? null;
  }, [remoteId]);

  useEffect(() => {
    syncMessagesFromServerRef.current = syncMessagesFromServer;
  }, [syncMessagesFromServer]);

  const latestUserMessageId =
    [...chat.messages]
      .reverse()
      .find((message) => message.role === "user")?.id ?? null;

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
    lastUpdatedUserMessageKeyRef.current = updateKey;
  }, [latestUserMessageId, remoteId, runtimeThreadKey]);

  const runtime = useAISDKRuntime({
    ...chat,
    stop,
  });

  if (transport instanceof AssistantChatTransport) {
    transport.setRuntime(assistantRuntime);
  }

  return runtime;
};

const ThreadStoreSync = () => {
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);

  useEffect(() => {
    if (!remoteId) {
      return;
    }

    setStoredActiveThreadId(remoteId);
  }, [remoteId]);

  return null;
};

const InitialThreadSwitch = ({
  storedThreadId,
  onResolved,
}: {
  storedThreadId?: string;
  onResolved: () => void;
}) => {
  const aui = useAui();
  const isLoading = useAuiState((s) => s.threads.isLoading);
  const mainThreadId = useAuiState((s) => s.threads.mainThreadId);
  const threadIds = useAuiState((s) => s.threads.threadIds);
  const hasSwitchedRef = useRef(false);

  useEffect(() => {
    if (isLoading) {
      return;
    }

    if (!storedThreadId) {
      onResolved();
      return;
    }

    if (mainThreadId === storedThreadId) {
      hasSwitchedRef.current = true;
      onResolved();
      return;
    }

    if (hasSwitchedRef.current) {
      onResolved();
      return;
    }

    if (!threadIds.includes(storedThreadId)) {
      hasSwitchedRef.current = true;
      clearStoredActiveThreadId();
      onResolved();
      return;
    }

    hasSwitchedRef.current = true;
    void (async () => {
      try {
        await aui.threads().switchToThread(storedThreadId);
      } catch {
        clearStoredActiveThreadId();
      } finally {
        onResolved();
      }
    })();
  }, [aui, isLoading, mainThreadId, onResolved, storedThreadId, threadIds]);

  return null;
};

const ChatShellRuntime = ({
  apiBaseUrl,
  storedThreadId,
}: {
  apiBaseUrl: string;
  storedThreadId?: string;
}) => {
  const [isInitialThreadResolved, setIsInitialThreadResolved] = useState(
    !storedThreadId,
  );

  useEffect(() => {
    setIsInitialThreadResolved(!storedThreadId);
  }, [storedThreadId]);

  const handleInitialThreadResolved = useCallback(() => {
    setIsInitialThreadResolved(true);
  }, []);

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
      <InitialThreadSwitch
        storedThreadId={storedThreadId}
        onResolved={handleInitialThreadResolved}
      />
      <ThreadStoreSync />
      <ChatShellLayout isInitialThreadResolved={isInitialThreadResolved} />
    </AssistantRuntimeProvider>
  );
};

const ChatShellLayout = ({
  isInitialThreadResolved,
}: {
  isInitialThreadResolved: boolean;
}) => {
  return (
    <SidebarProvider defaultOpen>
      <ThreadListSidebar />
      <SidebarInset className="bg-[#212121] text-[#ececec]">
        <div className="flex h-14 items-center border-b border-white/10 px-3">
          <SidebarTrigger className="text-[#bcbcbc] hover:bg-white/10 hover:text-[#f2f2f2]" />
        </div>
        <main className="min-h-0 flex-1">
          {!isInitialThreadResolved ? (
            <div className="flex h-full items-center justify-center text-sm text-[#9f9f9f]">
              Loading thread...
            </div>
          ) : (
            <Assistant />
          )}
        </main>
      </SidebarInset>
    </SidebarProvider>
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
