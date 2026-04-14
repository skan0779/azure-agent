import {
  ACTIVE_THREAD_ID_STORAGE_KEY,
  THREAD_STORE_STORAGE_KEY,
} from "@/lib/thread-store.keys";
import type {
  CreateThreadInput,
  ThreadId,
  ThreadStore,
  ThreadStoreState,
  ThreadSummary,
  UpdateThreadInput,
} from "@/lib/thread-store";

type LocalThreadStore = {
  getState(): ThreadStoreState;
  createThread(input?: CreateThreadInput): ThreadSummary;
  updateThread(threadId: ThreadId, patch: UpdateThreadInput): ThreadSummary;
  deleteThread(threadId: ThreadId): void;
  setActiveThread(threadId: ThreadId): void;
  clearActiveThread(): void;
};

const DEFAULT_THREAD_TITLE = "New chat";

const canUseStorage = () => typeof window !== "undefined";

const createThreadId = () => crypto.randomUUID();

const nowIso = () => new Date().toISOString();

const sortThreads = (threads: ThreadSummary[]) => {
  return [...threads].sort((a, b) => {
    return (
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  });
};

const readThreads = (): ThreadSummary[] => {
  if (!canUseStorage()) {
    return [];
  }

  const raw = window.localStorage.getItem(THREAD_STORE_STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(isThreadSummary);
  } catch {
    return [];
  }
};

const writeThreads = (threads: ThreadSummary[]) => {
  if (!canUseStorage()) {
    return;
  }

  window.localStorage.setItem(
    THREAD_STORE_STORAGE_KEY,
    JSON.stringify(sortThreads(threads)),
  );
};

const readActiveThreadId = (): ThreadId | null => {
  if (!canUseStorage()) {
    return null;
  }

  return window.localStorage.getItem(ACTIVE_THREAD_ID_STORAGE_KEY);
};

const writeActiveThreadId = (threadId: ThreadId | null) => {
  if (!canUseStorage()) {
    return;
  }

  if (!threadId) {
    window.localStorage.removeItem(ACTIVE_THREAD_ID_STORAGE_KEY);
    return;
  }

  window.localStorage.setItem(ACTIVE_THREAD_ID_STORAGE_KEY, threadId);
};

const isThreadSummary = (value: unknown): value is ThreadSummary => {
  if (!value || typeof value !== "object") {
    return false;
  }

  return (
    "id" in value &&
    typeof value.id === "string" &&
    "title" in value &&
    typeof value.title === "string" &&
    "createdAt" in value &&
    typeof value.createdAt === "string" &&
    "updatedAt" in value &&
    typeof value.updatedAt === "string"
  );
};

const createThreadSummary = (input: CreateThreadInput = {}): ThreadSummary => {
  const timestamp = nowIso();

  return {
    id: input.id ?? createThreadId(),
    title: input.title?.trim() || DEFAULT_THREAD_TITLE,
    createdAt: input.createdAt ?? timestamp,
    updatedAt: input.updatedAt ?? timestamp,
    preview: input.preview,
    lastJobId: input.lastJobId,
    titleSource: input.titleSource ?? "manual",
  };
};

export const localThreadStore: ThreadStore & LocalThreadStore = {
  getState(): ThreadStoreState {
    const threads = sortThreads(readThreads());
    const activeThreadId = readActiveThreadId();

    return {
      threads,
      activeThreadId:
        activeThreadId && threads.some((thread) => thread.id === activeThreadId)
          ? activeThreadId
          : null,
    };
  },

  createThread(input = {}): ThreadSummary {
    const thread = createThreadSummary(input);
    const threads = readThreads();
    const nextThreads = sortThreads([thread, ...threads]);

    writeThreads(nextThreads);
    writeActiveThreadId(thread.id);

    return thread;
  },

  updateThread(threadId, patch): ThreadSummary {
    const threads = readThreads();
    const index = threads.findIndex((thread) => thread.id === threadId);

    if (index === -1) {
      throw new Error(`Thread not found: ${threadId}`);
    }

    const current = threads[index];
    const next: ThreadSummary = {
      ...current,
      ...patch,
      title:
        patch.title !== undefined
          ? patch.title.trim() || DEFAULT_THREAD_TITLE
          : current.title,
      updatedAt: patch.updatedAt ?? nowIso(),
    };

    const nextThreads = [...threads];
    nextThreads[index] = next;
    writeThreads(nextThreads);

    return next;
  },

  deleteThread(threadId): void {
    const threads = readThreads();
    const nextThreads = threads.filter((thread) => thread.id !== threadId);
    writeThreads(nextThreads);

    if (readActiveThreadId() === threadId) {
      writeActiveThreadId(nextThreads[0]?.id ?? null);
    }
  },

  setActiveThread(threadId): void {
    writeActiveThreadId(threadId);
  },

  clearActiveThread(): void {
    writeActiveThreadId(null);
  },
};

export const ensureLocalActiveThread = (
  store: ThreadStore = localThreadStore,
): ThreadSummary => {
  const localStore = store as LocalThreadStore;
  const state = localStore.getState();

  if (state.activeThreadId) {
    const existingThread = state.threads.find(
      (thread) => thread.id === state.activeThreadId,
    );
    if (existingThread) {
      return existingThread;
    }
  }

  if (state.threads[0]) {
    localStore.setActiveThread(state.threads[0].id);
    return state.threads[0];
  }

  return localStore.createThread();
};
