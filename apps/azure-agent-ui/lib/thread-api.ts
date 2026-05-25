import type { UIMessage } from "ai";

import { buildBearerHeaders, type AccessTokenProvider } from "@/lib/auth";
import type { ThreadSummary, ThreadTitleSource } from "@/lib/thread-store";

const buildJsonHeaders = (getAccessToken: AccessTokenProvider) =>
  buildBearerHeaders(getAccessToken, {
    "Content-Type": "application/json",
  });

const parseError = async (response: Response) => {
  try {
    const payload = (await response.json()) as {
      error?: string;
      detail?: unknown;
    };
    if (payload.error) {
      return payload.error;
    }
  } catch {
    // Ignore JSON parsing failure.
  }

  return response.statusText || `HTTP ${response.status}`;
};

export const listThreads = async ({
  apiBaseUrl,
  getAccessToken,
}: {
  apiBaseUrl: string;
  getAccessToken: AccessTokenProvider;
}): Promise<ThreadSummary[]> => {
  const response = await fetch(`${apiBaseUrl}/api/threads`, {
    headers: await buildBearerHeaders(getAccessToken),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as ThreadSummary[];
};

export const updateThread = async ({
  apiBaseUrl,
  getAccessToken,
  threadId,
  title,
  titleSource,
  updatedAt,
  lastJobId,
}: {
  apiBaseUrl: string;
  getAccessToken: AccessTokenProvider;
  threadId: string;
  title?: string;
  titleSource?: ThreadTitleSource;
  updatedAt?: string;
  lastJobId?: string;
}): Promise<ThreadSummary> => {
  const response = await fetch(
    `${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}`,
    {
      method: "PATCH",
      headers: await buildJsonHeaders(getAccessToken),
      body: JSON.stringify({
        title,
        titleSource,
        updatedAt,
        lastJobId,
      }),
    },
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as ThreadSummary;
};

export const deleteThread = async ({
  apiBaseUrl,
  getAccessToken,
  threadId,
}: {
  apiBaseUrl: string;
  getAccessToken: AccessTokenProvider;
  threadId: string;
}): Promise<void> => {
  const response = await fetch(
    `${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}`,
    {
      method: "DELETE",
      headers: await buildBearerHeaders(getAccessToken),
    },
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }
};

export const getThreadMessages = async ({
  apiBaseUrl,
  getAccessToken,
  threadId,
}: {
  apiBaseUrl: string;
  getAccessToken: AccessTokenProvider;
  threadId: string;
}): Promise<UIMessage[]> => {
  const response = await fetch(
    `${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}/messages`,
    {
      headers: await buildBearerHeaders(getAccessToken),
    },
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as UIMessage[];
};
