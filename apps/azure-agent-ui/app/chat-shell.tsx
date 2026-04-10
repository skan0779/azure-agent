"use client";

import { startTransition, useEffect, useMemo, useState } from "react";

import { Assistant } from "@/app/assistant";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  type ChatSession,
  createChatSession,
  loadActiveChatId,
  loadSessions,
  loadUserId,
  saveActiveChatId,
  saveSessions,
  saveUserId,
} from "@/lib/chat-storage";

const ensureSessions = (
  sessions: ChatSession[],
  activeChatId: string | null,
): { sessions: ChatSession[]; activeChatId: string } => {
  const normalized = sessions.length > 0 ? sessions : [createChatSession()];
  const fallbackId = normalized[0]!.id;
  const nextActiveChatId =
    activeChatId && normalized.some((session) => session.id === activeChatId)
      ? activeChatId
      : fallbackId;

  return {
    sessions: normalized,
    activeChatId: nextActiveChatId,
  };
};

export function ChatShell() {
  const [isReady, setIsReady] = useState(false);
  const [userId, setUserId] = useState("");
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState("");

  useEffect(() => {
    const storedUserId = loadUserId();
    const storedSessions = loadSessions();
    const storedActiveChatId = loadActiveChatId();
    const nextState = ensureSessions(storedSessions, storedActiveChatId);

    startTransition(() => {
      setUserId(storedUserId);
      setSessions(nextState.sessions);
      setActiveChatId(nextState.activeChatId);
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
    if (!isReady) {
      return;
    }

    saveSessions(sessions);
  }, [isReady, sessions]);

  useEffect(() => {
    if (!isReady || !activeChatId) {
      return;
    }

    saveActiveChatId(activeChatId);
  }, [activeChatId, isReady]);

  const handleNewChat = () => {
    const session = createChatSession();

    setSessions((current) => [session, ...current]);
    setActiveChatId(session.id);
  };

  const handleSelectChat = (chatId: string) => {
    setActiveChatId(chatId);
  };

  const handleSessionTouched = (chatId: string, titleHint?: string) => {
    setSessions((current) => {
      return current.map((session) => {
        if (session.id !== chatId) {
          return session;
        }

        const normalizedTitle =
          titleHint?.trim() ||
          (session.title === "New Chat" ? session.title : session.title);

        return {
          ...session,
          title:
            session.title === "New Chat" && normalizedTitle !== "New Chat"
              ? normalizedTitle
              : session.title,
          updatedAt: new Date().toISOString(),
        };
      });
    });
  };

  const sidebarSessions = useMemo(() => {
    return [...sessions].sort((left, right) =>
      right.updatedAt.localeCompare(left.updatedAt),
    );
  }, [sessions]);

  if (!isReady || !activeChatId) {
    return null;
  }

  return (
    <div className="flex h-dvh bg-background">
      <ThreadListSidebar
        activeChatId={activeChatId}
        sessions={sidebarSessions}
        userId={userId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onUserIdChange={setUserId}
      />
      <main className="min-w-0 flex-1">
        <Assistant
          key={activeChatId}
          chatId={activeChatId}
          userId={userId}
          onSessionTouched={handleSessionTouched}
        />
      </main>
    </div>
  );
}
