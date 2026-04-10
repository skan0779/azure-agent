"use client";

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

const STORAGE_PREFIX = "azure-agent-ui";
const USER_ID_KEY = `${STORAGE_PREFIX}:user-id`;
const ACTIVE_CHAT_ID_KEY = `${STORAGE_PREFIX}:active-chat-id`;
const SESSIONS_KEY = `${STORAGE_PREFIX}:sessions`;
const THREAD_PREFIX = `${STORAGE_PREFIX}:thread:`;

export const createChatId = (): string => {
  return crypto.randomUUID();
};

export const createChatSession = (title = "New Chat"): ChatSession => {
  const timestamp = new Date().toISOString();

  return {
    id: createChatId(),
    title,
    createdAt: timestamp,
    updatedAt: timestamp,
  };
};

export const loadUserId = (): string => {
  if (typeof window === "undefined") {
    return "";
  }

  return window.localStorage.getItem(USER_ID_KEY) ?? "";
};

export const saveUserId = (userId: string): void => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(USER_ID_KEY, userId);
};

export const loadSessions = (): ChatSession[] => {
  if (typeof window === "undefined") {
    return [];
  }

  const raw = window.localStorage.getItem(SESSIONS_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as ChatSession[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
};

export const saveSessions = (sessions: ChatSession[]): void => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
};

export const loadActiveChatId = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(ACTIVE_CHAT_ID_KEY);
};

export const saveActiveChatId = (chatId: string): void => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(ACTIVE_CHAT_ID_KEY, chatId);
};

export const getThreadStorageKey = (chatId: string): string => {
  return `${THREAD_PREFIX}${chatId}`;
};
