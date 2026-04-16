import type { RemoteThreadListAdapter } from "@assistant-ui/react";

import {
  createThread,
  deleteThread,
  listThreads,
  updateThread,
} from "@/lib/thread-api";
import { DEFAULT_THREAD_TITLE } from "@/lib/thread-store.local";

type RemoteThreadInitializeResponse = {
  remoteId: string;
  externalId: string | undefined;
};

type RemoteThreadMetadata = {
  readonly status: "regular" | "archived";
  readonly remoteId: string;
  readonly externalId?: string | undefined;
  readonly title?: string | undefined;
};

type RemoteThreadListResponse = {
  threads: RemoteThreadMetadata[];
};

const EMPTY_TITLE_STREAM = new ReadableStream();

const toMetadata = (thread: {
  id: string;
  title: string;
}): RemoteThreadMetadata => ({
  status: "regular",
  remoteId: thread.id,
  externalId: undefined,
  title: thread.title,
});

export class RemoteApiThreadListAdapter implements RemoteThreadListAdapter {
  constructor(
    private readonly options: {
      apiBaseUrl: string;
      userId: string;
    },
  ) {}

  async list(): Promise<RemoteThreadListResponse> {
    const threads = await listThreads(this.options);

    return {
      threads: threads.map(toMetadata),
    };
  }

  async rename(remoteId: string, newTitle: string): Promise<void> {
    await updateThread({
      ...this.options,
      threadId: remoteId,
      title: newTitle,
      titleSource: "manual",
    });
  }

  async archive(): Promise<void> {
    return;
  }

  async unarchive(): Promise<void> {
    return;
  }

  async delete(remoteId: string): Promise<void> {
    await deleteThread({
      ...this.options,
      threadId: remoteId,
    });
  }

  async initialize(_threadId: string): Promise<RemoteThreadInitializeResponse> {
    void _threadId;

    const created = await createThread({
      ...this.options,
      title: DEFAULT_THREAD_TITLE,
      titleSource: "manual",
    });

    return {
      remoteId: created.id,
      externalId: undefined,
    };
  }

  async generateTitle() {
    return EMPTY_TITLE_STREAM;
  }

  async fetch(threadId: string): Promise<RemoteThreadMetadata> {
    const threads = await listThreads(this.options);
    const thread = threads.find((item) => item.id === threadId);

    if (!thread) {
      const created = await createThread({
        ...this.options,
        id: threadId,
        title: DEFAULT_THREAD_TITLE,
        titleSource: "manual",
      });
      return toMetadata(created);
    }

    return toMetadata(thread);
  }
}
