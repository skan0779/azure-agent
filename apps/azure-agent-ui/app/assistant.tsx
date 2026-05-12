"use client";

import { Thread } from "@/components/assistant-ui/thread";

export const Assistant = ({
  apiBaseUrl,
  userId,
}: {
  apiBaseUrl: string;
  userId: string;
}) => {
  return (
    <div className="h-full min-h-0 min-w-0 bg-[#212121]">
      <Thread apiBaseUrl={apiBaseUrl} userId={userId} />
    </div>
  );
};
