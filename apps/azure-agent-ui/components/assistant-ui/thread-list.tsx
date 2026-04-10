"use client";

import { MessageSquarePlus, PanelLeft } from "lucide-react";

import type { ChatSession } from "@/lib/chat-storage";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

type ThreadListProps = {
  activeChatId: string;
  sessions: ChatSession[];
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
};

const formatTimestamp = (value: string): string => {
  const date = new Date(value);
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
};

export function ThreadList({
  activeChatId,
  sessions,
  onNewChat,
  onSelectChat,
}: ThreadListProps) {
  return (
    <div className="flex h-full flex-col gap-3">
      <Button
        type="button"
        onClick={onNewChat}
        className="justify-start gap-2 rounded-2xl"
      >
        <MessageSquarePlus className="size-4" />
        New Chat
      </Button>

      <div className="flex items-center gap-2 px-1 text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
        <PanelLeft className="size-3.5" />
        Threads
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto pr-1">
        {sessions.map((session) => {
          const isActive = session.id === activeChatId;

          return (
            <button
              key={session.id}
              type="button"
              onClick={() => onSelectChat(session.id)}
              className={cn(
                "flex w-full flex-col items-start gap-1 rounded-2xl border px-3 py-3 text-left transition-colors",
                isActive
                  ? "border-foreground/10 bg-foreground text-background"
                  : "border-transparent bg-muted/60 hover:bg-muted",
              )}
            >
              <span
                className={cn(
                  "line-clamp-1 w-full text-sm font-medium",
                  isActive ? "text-background" : "text-foreground",
                )}
              >
                {session.title}
              </span>
              <span
                className={cn(
                  "text-xs",
                  isActive
                    ? "text-background/80"
                    : "text-muted-foreground",
                )}
              >
                {formatTimestamp(session.updatedAt)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
