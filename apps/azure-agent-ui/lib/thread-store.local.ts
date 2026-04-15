import {
  ACTIVE_THREAD_ID_STORAGE_KEY,
  THREAD_MESSAGES_STORAGE_KEY,
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

export const DEFAULT_THREAD_TITLE = "New chat";

const THREAD_ID_UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

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

export const isUuidThreadId = (
  threadId: string | null | undefined,
): threadId is ThreadId => {
  return typeof threadId === "string" && THREAD_ID_UUID_REGEX.test(threadId);
};

const normalizeThreadStorage = () => {
  if (!canUseStorage()) {
    return;
  }

  const rawThreads = window.localStorage.getItem(THREAD_STORE_STORAGE_KEY);
  const rawActiveThreadId = window.localStorage.getItem(
    ACTIVE_THREAD_ID_STORAGE_KEY,
  );
  const rawThreadMessages = window.localStorage.getItem(
    THREAD_MESSAGES_STORAGE_KEY,
  );

  const idMap = new Map<string, string>();
  let hasMutation = false;

  const normalizedThreads: ThreadSummary[] = [];
  if (rawThreads) {
    try {
      const parsed = JSON.parse(rawThreads);
      if (Array.isArray(parsed)) {
        for (const value of parsed) {
          if (!isThreadSummary(value)) {
            hasMutation = true;
            continue;
          }

          let nextId = value.id;
          if (!isUuidThreadId(nextId)) {
            nextId = idMap.get(value.id) ?? createThreadId();
            idMap.set(value.id, nextId);
            hasMutation = true;
          }

          normalizedThreads.push({
            ...value,
            id: nextId,
          });
        }
      } else {
        hasMutation = true;
      }
    } catch {
      hasMutation = true;
    }
  }

  const uniqueThreads = sortThreads(
    Array.from(new Map(normalizedThreads.map((thread) => [thread.id, thread])).values()),
  );

  let nextActiveThreadId: string | null = rawActiveThreadId;
  if (nextActiveThreadId && idMap.has(nextActiveThreadId)) {
    nextActiveThreadId = idMap.get(nextActiveThreadId) ?? null;
    hasMutation = true;
  }

  if (
    nextActiveThreadId &&
    (!isUuidThreadId(nextActiveThreadId) ||
      !uniqueThreads.some((thread) => thread.id === nextActiveThreadId))
  ) {
    nextActiveThreadId = uniqueThreads[0]?.id ?? null;
    hasMutation = true;
  }

  const normalizedThreadMessages: Record<string, unknown> = {};
  if (rawThreadMessages) {
    try {
      const parsed = JSON.parse(rawThreadMessages);
      if (parsed && typeof parsed === "object") {
        for (const [threadId, messages] of Object.entries(parsed)) {
          const nextThreadId = isUuidThreadId(threadId)
            ? threadId
            : (idMap.get(threadId) ?? null);

          if (!nextThreadId) {
            hasMutation = true;
            continue;
          }

          normalizedThreadMessages[nextThreadId] = messages;
          if (nextThreadId !== threadId) {
            hasMutation = true;
          }
        }
      } else {
        hasMutation = true;
      }
    } catch {
      hasMutation = true;
    }
  }

  if (!hasMutation) {
    return;
  }

  window.localStorage.setItem(
    THREAD_STORE_STORAGE_KEY,
    JSON.stringify(uniqueThreads),
  );

  if (nextActiveThreadId) {
    window.localStorage.setItem(
      ACTIVE_THREAD_ID_STORAGE_KEY,
      nextActiveThreadId,
    );
  } else {
    window.localStorage.removeItem(ACTIVE_THREAD_ID_STORAGE_KEY);
  }

  window.localStorage.setItem(
    THREAD_MESSAGES_STORAGE_KEY,
    JSON.stringify(normalizedThreadMessages),
  );
};

const readThreads = (): ThreadSummary[] => {
  if (!canUseStorage()) {
    return [];
  }

  normalizeThreadStorage();

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

  normalizeThreadStorage();

  const activeThreadId = window.localStorage.getItem(ACTIVE_THREAD_ID_STORAGE_KEY);
  return isUuidThreadId(activeThreadId) ? activeThreadId : null;
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
    id: isUuidThreadId(input.id) ? input.id : createThreadId(),
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
