import type { RemoteThreadListAdapter } from "@assistant-ui/react";

import {
  deleteThread,
  listThreads,
  updateThread,
} from "@/lib/thread-api";

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

type GenerateTitleStream = Awaited<
  ReturnType<RemoteThreadListAdapter["generateTitle"]>
>;

const createEmptyTitleStream = (): GenerateTitleStream =>
  new ReadableStream({
    start(controller) {
      controller.close();
    },
  }) as GenerateTitleStream;

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

    return {
      remoteId: crypto.randomUUID(),
      externalId: undefined,
    };
  }

  async generateTitle() {
    return createEmptyTitleStream();
  }

  async fetch(threadId: string): Promise<RemoteThreadMetadata> {
    const threads = await listThreads(this.options);
    const thread = threads.find((item) => item.id === threadId);

    if (!thread) {
      throw new Error(`Thread not found: ${threadId}`);
    }

    return toMetadata(thread);
  }
}
