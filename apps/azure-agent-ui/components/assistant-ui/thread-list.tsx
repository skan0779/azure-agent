"use client";

import { type FC, useEffect, useRef, useState, useSyncExternalStore } from "react";
import {
  AuiIf,
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  EllipsisIcon,
  MessageSquareText,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";

import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useSidebar } from "@/components/ui/sidebar";
import {
  clearOptimisticThreadTitle,
  getOptimisticThreadTitle,
  subscribeOptimisticThreadTitles,
} from "@/lib/optimistic-thread-titles";

export const ThreadList: FC = () => {
  const { isMobile, setOpenMobile } = useSidebar();

  return (
    <ThreadListPrimitive.Root className="flex h-full flex-col">
      <ThreadListPrimitive.New
        className="flex w-full items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-left text-sm font-medium text-[#ececec] transition hover:bg-white/10 data-[active=true]:bg-white/10"
        onClick={() => {
          if (isMobile) {
            setOpenMobile(false);
          }
        }}
      >
        <PlusIcon className="size-4 shrink-0" />
        <span>New chat</span>
      </ThreadListPrimitive.New>
      <div className="aui-scrollbar mt-4 flex-1 overflow-y-auto">
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
  const { isMobile, setOpenMobile } = useSidebar();
  const title = useAuiState((s) => s.threadListItem.title ?? "New chat");
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const status = useAuiState((s) => s.threadListItem.status);
  const isMain = useAuiState((s) => s.threads.mainThreadId === s.threadListItem.id);
  const [isEditing, setIsEditing] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const pendingRenameFocusRef = useRef(false);
  const optimisticTitle = useSyncExternalStore(
    subscribeOptimisticThreadTitles,
    () => getOptimisticThreadTitle(remoteId),
    () => undefined,
  );
  const displayTitle = title === "New chat" && optimisticTitle ? optimisticTitle : title;
  const [draftTitle, setDraftTitle] = useState(displayTitle);

  const canRename = status !== "new";

  if (status === "new" && !isMain) {
    return null;
  }

  const commitRename = () => {
    const nextTitle = draftTitle.trim();
    setIsEditing(false);

    if (!canRename || !nextTitle || nextTitle === displayTitle) {
      setDraftTitle(displayTitle);
      return;
    }

    aui.threadListItem().rename(nextTitle);
  };

  useEffect(() => {
    setDraftTitle(displayTitle);
  }, [displayTitle]);

  useEffect(() => {
    if (!remoteId || title === "New chat") {
      return;
    }

    clearOptimisticThreadTitle(remoteId);
  }, [remoteId, title]);

  useEffect(() => {
    if (!isEditing) {
      return;
    }

    const frame = requestAnimationFrame(() => {
      const input = inputRef.current;
      if (!input) {
        return;
      }

      input.focus();
      const end = input.value.length;
      input.setSelectionRange(end, end);
      pendingRenameFocusRef.current = false;
    });

    return () => {
      cancelAnimationFrame(frame);
    };
  }, [isEditing]);

  return (
    <ThreadListItemPrimitive.Root className="group/thread-item">
      <div className="flex items-center gap-1">
        <ThreadListItemPrimitive.Trigger
          className={cn(
            "flex min-w-0 flex-1 items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-[#cdcdcd] transition hover:bg-white/5 hover:text-[#f2f2f2] data-[active=true]:bg-white/10 data-[active=true]:text-[#f8f8f8]",
            isEditing && "bg-white/10 text-[#f8f8f8]",
          )}
          onClick={() => {
            if (isMobile) {
              setOpenMobile(false);
            }
          }}
        >
          <MessageSquareText className="size-4 shrink-0 opacity-70" />
          <div
            className={cn(
              "min-w-0 flex-1 truncate",
              "group-data-[active=true]/thread-item:text-[#f8f8f8]",
            )}
          >
            {isEditing ? (
              <input
                ref={inputRef}
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
                    setDraftTitle(displayTitle);
                    setIsEditing(false);
                  }
                }}
                className="w-full rounded-md bg-transparent text-[#f8f8f8] outline-none placeholder:text-[#7f7f7f] caret-[#f8f8f8]"
                aria-label="Rename thread"
              />
            ) : (
              <span className="truncate">{displayTitle}</span>
            )}
          </div>
        </ThreadListItemPrimitive.Trigger>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              onClick={(event) => event.stopPropagation()}
              className="flex size-8 shrink-0 items-center justify-center rounded-lg text-[#9f9f9f] opacity-100 transition hover:bg-white/10 hover:text-[#f3f3f3] md:opacity-0 md:group-hover/thread-item:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100"
              aria-label="Thread actions"
            >
              <EllipsisIcon className="size-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            side="bottom"
            onCloseAutoFocus={(event) => {
              if (pendingRenameFocusRef.current || isEditing) {
                event.preventDefault();
              }
            }}
          >
            <DropdownMenuItem
              disabled={!canRename}
              onSelect={() => {
                if (!canRename) {
                  return;
                }

                pendingRenameFocusRef.current = true;
                setDraftTitle(displayTitle);
                setIsEditing(true);
              }}
            >
              <PencilIcon className="size-4" />
              <span>Rename</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-[#ffb4b4] focus:bg-red-500/15 focus:text-[#ffd0d0]"
              onSelect={() => {
                void aui.threadListItem().delete();
              }}
            >
              <Trash2Icon className="size-4" />
              <span>Delete</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </ThreadListItemPrimitive.Root>
  );
};
