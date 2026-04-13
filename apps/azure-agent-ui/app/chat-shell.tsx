"use client";

import { startTransition, useCallback, useEffect, useState } from "react";
import { Bot } from "lucide-react";

import { Assistant } from "@/app/assistant";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  createThreadId,
  loadThreadId,
  loadUserId,
  saveThreadId,
  saveUserId,
} from "@/lib/chat-storage";

export function ChatShell() {
  const [isReady, setIsReady] = useState(false);
  const [userId, setUserId] = useState("");
  const [threadId, setThreadId] = useState("");

  useEffect(() => {
    const storedUserId = loadUserId();
    const storedThreadId = loadThreadId() ?? createThreadId();

    startTransition(() => {
      setUserId(storedUserId);
      setThreadId(storedThreadId);
      setIsReady(true);
    });
  }, []);

  useEffect(() => {
    if (!isReady) {
      return;
    }

    saveUserId(userId);
  }, [isReady, userId]);

  useEffect(() => {
    if (!isReady || !threadId) {
      return;
    }

    saveThreadId(threadId);
  }, [isReady, threadId]);

  const handleNewChat = useCallback(() => {
    setThreadId(createThreadId());
  }, []);

  if (!isReady || !threadId) {
    return null;
  }

  return (
    <div className="flex h-dvh flex-col bg-background">
      <header className="border-b bg-muted/20 px-4 py-4">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-2xl bg-foreground text-background">
              <Bot className="size-5" />
            </div>
            <div>
              <p className="text-sm font-semibold">Azure Agent</p>
              <p className="text-xs text-muted-foreground">
                assistant-ui thread client
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="min-w-0 md:w-72">
              <label
                htmlFor="dev-user-id"
                className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground"
              >
                Dev User ID
              </label>
              <Input
                id="dev-user-id"
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="user-123"
                autoComplete="off"
              />
            </div>

            <div className="min-w-0 md:w-80">
              <label
                htmlFor="thread-id"
                className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground"
              >
                Thread ID
              </label>
              <Input id="thread-id" value={threadId} readOnly />
            </div>

            <div className="pt-0 md:self-end">
              <Button onClick={handleNewChat}>New Chat</Button>
            </div>
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1">
        <Assistant threadId={threadId} userId={userId} />
      </main>
    </div>
  );
}
