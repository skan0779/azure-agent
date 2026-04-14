"use client";

import { type FC, useState } from "react";
import {
  AuiIf,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  MessageSquareText,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";

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
  const aui = useAui();
  const title = useAuiState((s) => s.threadListItem.title ?? "New chat");
  const status = useAuiState((s) => s.threadListItem.status);
  const isMain = useAuiState((s) => s.threads.mainThreadId === s.threadListItem.id);
  const [isEditing, setIsEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(title);

  const canRename = status !== "new";

  if (status === "new" && !isMain) {
    return null;
  }

  const commitRename = () => {
    const nextTitle = draftTitle.trim();
    setIsEditing(false);

    if (!canRename || !nextTitle || nextTitle === title) {
      setDraftTitle(title);
      return;
    }

    aui.threadListItem().rename(nextTitle);
  };

  return (
    <ThreadListItemPrimitive.Root className="group/thread-item">
      <div className="flex items-center gap-1">
        <ThreadListItemPrimitive.Trigger className="flex min-w-0 flex-1 items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-[#cdcdcd] transition hover:bg-white/5 hover:text-[#f2f2f2] data-[active=true]:bg-white/10 data-[active=true]:text-[#f8f8f8]">
          <MessageSquareText className="size-4 shrink-0 opacity-70" />
          <div
            className={cn(
              "min-w-0 flex-1 truncate",
              "group-data-[active=true]/thread-item:text-[#f8f8f8]",
            )}
          >
            {isEditing ? (
              <input
                autoFocus
                value={draftTitle}
                onChange={(event) => setDraftTitle(event.target.value)}
                onBlur={commitRename}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    commitRename();
                    return;
                  }

                  if (event.key === "Escape") {
                    event.preventDefault();
                    setDraftTitle(title);
                    setIsEditing(false);
                  }
                }}
                className="w-full rounded-md bg-transparent text-[#f8f8f8] outline-none placeholder:text-[#7f7f7f]"
                aria-label="Rename thread"
              />
            ) : (
              <ThreadListItemPrimitive.Title fallback="New chat" />
            )}
          </div>
        </ThreadListItemPrimitive.Trigger>
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            if (!canRename) {
              return;
            }

            setDraftTitle(title);
            setIsEditing(true);
          }}
          disabled={!canRename}
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-[#9f9f9f] opacity-0 transition hover:bg-white/10 hover:text-[#f3f3f3] group-hover/thread-item:opacity-100 focus-visible:opacity-100 disabled:pointer-events-none disabled:opacity-0"
          aria-label="Rename thread"
        >
          <PencilIcon className="size-4" />
        </button>
        <ThreadListItemPrimitive.Delete
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-[#9f9f9f] opacity-0 transition hover:bg-white/10 hover:text-[#f3f3f3] group-hover/thread-item:opacity-100 focus-visible:opacity-100"
          aria-label="Delete thread"
        >
          <Trash2Icon className="size-4" />
        </ThreadListItemPrimitive.Delete>
      </div>
    </ThreadListItemPrimitive.Root>
  );
};
