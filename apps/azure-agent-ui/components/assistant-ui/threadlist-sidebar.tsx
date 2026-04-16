"use client";

import type { FC } from "react";

import { ThreadList } from "@/components/assistant-ui/thread-list";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
} from "@/components/ui/sidebar";

export const ThreadListSidebar: FC = () => {
  return (
    <Sidebar>
      <SidebarHeader className="px-5 pb-3 pt-4">
        <div className="text-sm font-medium text-[#ececec]">azure-agent</div>
        <div className="mt-1 text-xs text-[#9f9f9f]">Thread history</div>
      </SidebarHeader>
      <SidebarContent className="p-3 pt-0">
        <ThreadList />
      </SidebarContent>
    </Sidebar>
  );
};
