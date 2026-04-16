import type { UIMessage } from "ai";

import type { ThreadSummary, ThreadTitleSource } from "@/lib/thread-store";

const buildHeaders = (userId: string) => ({
  "Content-Type": "application/json",
  "X-User-Id": userId,
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
  userId,
}: {
  apiBaseUrl: string;
  userId: string;
}): Promise<ThreadSummary[]> => {
  const response = await fetch(`${apiBaseUrl}/api/threads`, {
    headers: {
      "X-User-Id": userId,
    },
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as ThreadSummary[];
};

export const updateThread = async ({
  apiBaseUrl,
  userId,
  threadId,
  title,
  titleSource,
  updatedAt,
  lastJobId,
}: {
  apiBaseUrl: string;
  userId: string;
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
      headers: buildHeaders(userId),
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
  userId,
  threadId,
}: {
  apiBaseUrl: string;
  userId: string;
  threadId: string;
}): Promise<void> => {
  const response = await fetch(
    `${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}`,
    {
      method: "DELETE",
      headers: {
        "X-User-Id": userId,
      },
    },
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }
};

export const getThreadMessages = async ({
  apiBaseUrl,
  userId,
  threadId,
}: {
  apiBaseUrl: string;
  userId: string;
  threadId: string;
}): Promise<UIMessage[]> => {
  const response = await fetch(
    `${apiBaseUrl}/api/threads/${encodeURIComponent(threadId)}/messages`,
    {
      headers: {
        "X-User-Id": userId,
      },
    },
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as UIMessage[];
};
