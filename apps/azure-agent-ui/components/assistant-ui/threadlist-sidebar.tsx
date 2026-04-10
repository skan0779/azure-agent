"use client";

import { Bot } from "lucide-react";

import type { ChatSession } from "@/lib/chat-storage";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { Input } from "@/components/ui/input";

type ThreadListSidebarProps = {
  activeChatId: string;
  sessions: ChatSession[];
  userId: string;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onUserIdChange: (userId: string) => void;
};

export function ThreadListSidebar({
  activeChatId,
  sessions,
  userId,
  onNewChat,
  onSelectChat,
  onUserIdChange,
}: ThreadListSidebarProps) {
  return (
    <aside className="flex h-full w-full max-w-72 flex-col border-r bg-muted/20">
      <div className="border-b px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="flex size-10 items-center justify-center rounded-2xl bg-foreground text-background">
            <Bot className="size-5" />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold">Azure Agent</p>
            <p className="text-xs text-muted-foreground">assistant-ui</p>
          </div>
        </div>
      </div>

      <div className="border-b px-4 py-4">
        <label
          htmlFor="dev-user-id"
          className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground"
        >
          Dev User ID
        </label>
        <Input
          id="dev-user-id"
          value={userId}
          onChange={(event) => onUserIdChange(event.target.value)}
          placeholder="user-123"
          autoComplete="off"
        />
      </div>

      <div className="min-h-0 flex-1 px-3 py-3">
        <ThreadList
          activeChatId={activeChatId}
          sessions={sessions}
          onNewChat={onNewChat}
          onSelectChat={onSelectChat}
        />
      </div>
    </aside>
  );
}
