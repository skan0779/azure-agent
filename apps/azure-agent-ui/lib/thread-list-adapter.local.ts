import type {
  RemoteThreadListAdapter,
} from "@assistant-ui/react";

import { localThreadStore } from "@/lib/thread-store.local";

type LocalRemoteThreadInitializeResponse = {
  remoteId: string;
  externalId: string | undefined;
};

type LocalRemoteThreadMetadata = {
  readonly status: "regular" | "archived";
  readonly remoteId: string;
  readonly externalId?: string | undefined;
  readonly title?: string | undefined;
};

type LocalRemoteThreadListResponse = {
  threads: LocalRemoteThreadMetadata[];
};

const EMPTY_TITLE_STREAM = new ReadableStream();

const getLocalThreadStoreState = () => {
  return localThreadStore.getState() as Awaited<
    ReturnType<typeof localThreadStore.getState>
  >;
};

const toRemoteThreadMetadata = (
  threadId: string,
): LocalRemoteThreadMetadata | undefined => {
  const state = getLocalThreadStoreState();
  const thread = state.threads.find((item) => item.id === threadId);

  if (!thread) {
    return undefined;
  }

  return {
    status: "regular",
    remoteId: thread.id,
    externalId: undefined,
    title: thread.title,
  };
};

export class LocalThreadListAdapter implements RemoteThreadListAdapter {
  async list(): Promise<LocalRemoteThreadListResponse> {
    const state = getLocalThreadStoreState();

    return {
      threads: state.threads.map((thread) => ({
        status: "regular" as const,
        remoteId: thread.id,
        externalId: undefined,
        title: thread.title,
      })),
    };
  }

  async rename(remoteId: string, newTitle: string): Promise<void> {
    localThreadStore.updateThread(remoteId, {
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
    localThreadStore.deleteThread(remoteId);
  }

  async initialize(
    threadId: string,
  ): Promise<LocalRemoteThreadInitializeResponse> {
    const existing = toRemoteThreadMetadata(threadId);

    if (!existing) {
      localThreadStore.createThread({ id: threadId });
    } else {
      localThreadStore.setActiveThread(threadId);
    }

    return {
      remoteId: threadId,
      externalId: undefined,
    };
  }

  async generateTitle() {
    return EMPTY_TITLE_STREAM;
  }

  async fetch(threadId: string): Promise<LocalRemoteThreadMetadata> {
    const thread = toRemoteThreadMetadata(threadId);

    if (thread) {
      return thread;
    }

    localThreadStore.createThread({ id: threadId });

    return {
      status: "regular",
      remoteId: threadId,
      externalId: undefined,
      title: "New chat",
    };
  }
}
