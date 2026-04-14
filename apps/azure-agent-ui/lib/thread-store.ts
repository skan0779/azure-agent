export type MaybePromise<T> = T | Promise<T>;

export type ThreadId = string;

export type ThreadTitleSource = "manual" | "first-user-message" | "generated";

export interface ThreadSummary {
  id: ThreadId;
  title: string;
  createdAt: string;
  updatedAt: string;
  preview?: string;
  lastJobId?: string;
  titleSource?: ThreadTitleSource;
}

export interface ThreadStoreState {
  threads: ThreadSummary[];
  activeThreadId: ThreadId | null;
}

export interface CreateThreadInput {
  id?: ThreadId;
  title?: string;
  preview?: string;
  createdAt?: string;
  updatedAt?: string;
  lastJobId?: string;
  titleSource?: ThreadTitleSource;
}

export interface UpdateThreadInput {
  title?: string;
  preview?: string;
  updatedAt?: string;
  lastJobId?: string;
  titleSource?: ThreadTitleSource;
}

export interface ThreadStore {
  getState(): MaybePromise<ThreadStoreState>;
  createThread(input?: CreateThreadInput): MaybePromise<ThreadSummary>;
  updateThread(
    threadId: ThreadId,
    patch: UpdateThreadInput,
  ): MaybePromise<ThreadSummary>;
  deleteThread(threadId: ThreadId): MaybePromise<void>;
  setActiveThread(threadId: ThreadId): MaybePromise<void>;
  clearActiveThread(): MaybePromise<void>;
}
