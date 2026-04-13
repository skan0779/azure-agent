"use client";

const STORAGE_PREFIX = "azure-agent-ui";
const USER_ID_KEY = `${STORAGE_PREFIX}:user-id`;
const THREAD_ID_KEY = `${STORAGE_PREFIX}:thread-id`;

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

export const createThreadId = (): string => {
  return crypto.randomUUID();
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

export const loadThreadId = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(THREAD_ID_KEY);
};

export const saveThreadId = (threadId: string): void => {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(THREAD_ID_KEY, threadId);
};
