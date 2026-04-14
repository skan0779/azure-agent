"use client";

import type { FC } from "react";
import {
  AuiIf,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
} from "@assistant-ui/react";
import { MessageSquareText, PlusIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export const ThreadList: FC = () => {
  return (
    <ThreadListPrimitive.Root className="flex h-full flex-col">
      <ThreadListPrimitive.New className="flex w-full items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-left text-sm font-medium text-[#ececec] transition hover:bg-white/10 data-[active=true]:bg-white/10">
        <PlusIcon className="size-4 shrink-0" />
        <span>New chat</span>
      </ThreadListPrimitive.New>
      <div className="mt-4 flex-1 overflow-y-auto">
        <AuiIf condition={(s) => s.threads.isLoading}>
          <ThreadListSkeleton />
        </AuiIf>
        <AuiIf condition={(s) => !s.threads.isLoading}>
          <div className="space-y-1">
            <ThreadListPrimitive.Items components={{ ThreadListItem }} />
          </div>
        </AuiIf>
      </div>
    </ThreadListPrimitive.Root>
  );
};

const ThreadListSkeleton: FC = () => {
  return (
    <div className="space-y-1">
      {Array.from({ length: 6 }, (_, index) => (
        <div
          key={index}
          className="h-10 rounded-xl bg-white/5"
        />
      ))}
    </div>
  );
};

const ThreadListItem: FC = () => {
  return (
    <ThreadListItemPrimitive.Root className="group/thread-item">
      <ThreadListItemPrimitive.Trigger className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-[#cdcdcd] transition hover:bg-white/5 hover:text-[#f2f2f2] data-[active=true]:bg-white/10 data-[active=true]:text-[#f8f8f8]">
        <MessageSquareText className="size-4 shrink-0 opacity-70" />
        <div
          className={cn(
            "min-w-0 flex-1 truncate",
            "group-data-[active=true]/thread-item:text-[#f8f8f8]",
          )}
        >
          <ThreadListItemPrimitive.Title fallback="New chat" />
        </div>
      </ThreadListItemPrimitive.Trigger>
    </ThreadListItemPrimitive.Root>
  );
};
