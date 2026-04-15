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
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from "react";

import { Assistant } from "@/app/assistant";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  getThreadMessages,
  updateThread as updateRemoteThread,
} from "@/lib/thread-api";
import { localThreadMessageStore } from "@/lib/thread-message-store.local";
import { RemoteApiThreadListAdapter } from "@/lib/thread-list-adapter.remote";
import {
  DEFAULT_THREAD_TITLE,
  localThreadStore,
} from "@/lib/thread-store.local";

const DEFAULT_USER_ID = "1015520";
const AUTO_TITLE_TOKEN_LIMIT = 7;

const getLocalThreadStoreState = () => {
  return localThreadStore.getState() as Awaited<
    ReturnType<typeof localThreadStore.getState>
  >;
};

const getStoredActiveThreadId = () => {
  const state = getLocalThreadStoreState();

  if (state.activeThreadId) {
    const activeThread = state.threads.find(
      (thread) => thread.id === state.activeThreadId,
    );
    if (activeThread) {
      return activeThread.id;
    }
  }

  if (state.threads[0]) {
    localThreadStore.setActiveThread(state.threads[0].id);
    return state.threads[0].id;
  }

  return undefined;
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
  const threadId = useAuiState((s) => s.threadListItem.id);
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const storageThreadId = remoteId ?? threadId;
  const runtimeThreadKey = remoteId ?? threadId;
  const initialMessages = useMemo(
    () => localThreadMessageStore.getMessages(storageThreadId),
    [storageThreadId],
  );
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
    messages: initialMessages,
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
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

      if (!remoteId) {
        return;
      }

      void getThreadMessages({
        apiBaseUrl,
        userId,
        threadId: remoteId,
      })
        .then((messages) => {
          localThreadMessageStore.setMessages(remoteId, messages);
          chat.setMessages(messages);
        })
        .catch((error) => {
          console.warn("Failed to refresh thread messages after finish", {
            threadId: remoteId,
            error,
          });
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

  const latestUserMessageId =
    [...chat.messages]
      .reverse()
      .find((message) => message.role === "user")?.id ?? null;

  useEffect(() => {
    localThreadMessageStore.setMessages(storageThreadId, chat.messages);
  }, [chat.messages, storageThreadId]);

  useEffect(() => {
    if (!remoteId) {
      return;
    }

    let cancelled = false;

    void getThreadMessages({
      apiBaseUrl,
      userId,
      threadId: remoteId,
    })
      .then((messages) => {
        if (cancelled) {
          return;
        }

        localThreadMessageStore.setMessages(remoteId, messages);
        chat.setMessages(messages);
      })
      .catch((error) => {
        console.warn("Failed to load thread messages", {
          threadId: remoteId,
          error,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, chat, remoteId, userId]);

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
    localThreadStore.updateThread(remoteId, {
      updatedAt,
    });
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

    const state = getLocalThreadStoreState();
    const existing = state.threads.find((thread) => thread.id === remoteId);

    if (existing) {
      localThreadStore.setActiveThread(remoteId);
    }
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

    const state = getLocalThreadStoreState();
    if (!state.threads.some((thread) => thread.id === storedThreadId)) {
      hasSwitchedRef.current = true;
      return;
    }

    hasSwitchedRef.current = true;
    void aui.threads().switchToThread(storedThreadId);
  }, [aui, isLoading, mainThreadId, storedThreadId]);

  return null;
};

const ThreadTitleSync = ({
  apiBaseUrl,
}: {
  apiBaseUrl: string;
}) => {
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
      void updateRemoteThread({
        apiBaseUrl,
        userId: DEFAULT_USER_ID,
        threadId: remoteId,
        titleSource: "first-user-message",
      }).catch((error) => {
        console.warn("Failed to sync thread title source", {
          threadId: remoteId,
          error,
        });
      });
      return;
    }

    aui.threads().item("main").rename(firstUserTitle);
    localThreadStore.updateThread(remoteId, {
      title: firstUserTitle,
      titleSource: "first-user-message",
    });
    void updateRemoteThread({
      apiBaseUrl,
      userId: DEFAULT_USER_ID,
      threadId: remoteId,
      title: firstUserTitle,
      titleSource: "first-user-message",
    }).catch((error) => {
      console.warn("Failed to update thread title", {
        threadId: remoteId,
        error,
      });
    });
  }, [apiBaseUrl, aui, firstUserTitle, remoteId]);

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
      <ThreadTitleSync apiBaseUrl={apiBaseUrl} />
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
