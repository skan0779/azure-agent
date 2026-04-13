"use client";

import { startTransition, useEffect, useState } from "react";

import { Assistant } from "@/app/assistant";

const THREAD_ID_STORAGE_KEY = "azure-agent-ui:thread-id";
const DEFAULT_USER_ID = "1015520";

const createThreadId = () => crypto.randomUUID();

export default function Home() {
  const [threadId, setThreadId] = useState("");

  useEffect(() => {
    const storedThreadId =
      window.localStorage.getItem(THREAD_ID_STORAGE_KEY) ?? createThreadId();

    startTransition(() => {
      setThreadId(storedThreadId);
    });

    window.localStorage.setItem(THREAD_ID_STORAGE_KEY, storedThreadId);
  }, []);

  if (!threadId) {
    return null;
  }

  return <Assistant threadId={threadId} userId={DEFAULT_USER_ID} />;
}
