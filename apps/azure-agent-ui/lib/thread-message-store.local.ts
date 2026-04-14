import type { UIMessage } from "ai";

import { THREAD_MESSAGES_STORAGE_KEY } from "@/lib/thread-store.keys";

type ThreadMessagesMap = Record<string, UIMessage[]>;

const canUseStorage = () => typeof window !== "undefined";

const isUIMessage = (value: unknown): value is UIMessage => {
  if (!value || typeof value !== "object") {
    return false;
  }

  return (
    "id" in value &&
    typeof value.id === "string" &&
    "role" in value &&
    typeof value.role === "string" &&
    "parts" in value &&
    Array.isArray(value.parts)
  );
};

const readThreadMessagesMap = (): ThreadMessagesMap => {
  if (!canUseStorage()) {
    return {};
  }

  const raw = window.localStorage.getItem(THREAD_MESSAGES_STORAGE_KEY);
  if (!raw) {
    return {};
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {};
    }

    return Object.fromEntries(
      Object.entries(parsed).map(([threadId, messages]) => [
        threadId,
        Array.isArray(messages) ? messages.filter(isUIMessage) : [],
      ]),
    );
  } catch {
    return {};
  }
};

const writeThreadMessagesMap = (threadMessagesMap: ThreadMessagesMap) => {
  if (!canUseStorage()) {
    return;
  }

  window.localStorage.setItem(
    THREAD_MESSAGES_STORAGE_KEY,
    JSON.stringify(threadMessagesMap),
  );
};

export const localThreadMessageStore = {
  getMessages(threadId: string): UIMessage[] {
    const threadMessagesMap = readThreadMessagesMap();
    return threadMessagesMap[threadId] ?? [];
  },

  setMessages(threadId: string, messages: UIMessage[]) {
    const threadMessagesMap = readThreadMessagesMap();

    if (messages.length === 0) {
      delete threadMessagesMap[threadId];
      writeThreadMessagesMap(threadMessagesMap);
      return;
    }

    threadMessagesMap[threadId] = messages;
    writeThreadMessagesMap(threadMessagesMap);
  },

  clearMessages(threadId: string) {
    const threadMessagesMap = readThreadMessagesMap();
    delete threadMessagesMap[threadId];
    writeThreadMessagesMap(threadMessagesMap);
  },
};
