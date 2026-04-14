"use client";

import type { FC } from "react";

import { ThreadList } from "@/components/assistant-ui/thread-list";

export const ThreadListSidebar: FC = () => {
  return (
    <aside className="hidden h-dvh w-72 shrink-0 border-r border-white/10 bg-black/20 md:flex">
      <div className="flex h-full w-full flex-col p-3">
        <div className="px-2 pb-3 pt-1">
          <div className="text-sm font-medium text-[#ececec]">azure-agent</div>
          <div className="mt-1 text-xs text-[#9f9f9f]">
            Local thread history
          </div>
        </div>
        <ThreadList />
      </div>
    </aside>
  );
};
