"use client";

import { Thread } from "@/components/assistant-ui/thread";
import type { AccessTokenProvider } from "@/lib/auth";

export const Assistant = ({
  apiBaseUrl,
  getAccessToken,
}: {
  apiBaseUrl: string;
  getAccessToken: AccessTokenProvider;
}) => {
  return (
    <div className="h-full min-h-0 min-w-0 bg-[#212121]">
      <Thread apiBaseUrl={apiBaseUrl} getAccessToken={getAccessToken} />
    </div>
  );
};
