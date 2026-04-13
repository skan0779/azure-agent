import { parseSseStream } from "./sse.js";

export type AgentJobCreateResponse = {
  job_id: string;
  status: string;
  status_url: string;
  events_url: string;
  cancel_url: string;
};

export type AgentEvent = {
  type: string;
  ns?: string[];
  data?: unknown;
  event_id?: string;
};

const buildUrl = (baseUrl: string, path: string): string => {
  return `${baseUrl.replace(/\/$/, "")}${path}`;
};

const parseErrorMessage = async (response: Response): Promise<string> => {
  try {
    const payload = (await response.json()) as {
      detail?:
        | string
        | {
            message?: string;
            code?: string;
          };
    };

    if (typeof payload.detail === "string" && payload.detail) {
      return payload.detail;
    }

    if (
      payload.detail &&
      typeof payload.detail === "object" &&
      typeof payload.detail.message === "string" &&
      payload.detail.message
    ) {
      return payload.detail.message;
    }
  } catch {
    // Ignore JSON parsing failure and fall through to status text.
  }

  return response.statusText || `HTTP ${response.status}`;
};

export const createAgentJob = async ({
  baseUrl,
  threadId,
  userId,
  userQuery,
  signal,
}: {
  baseUrl: string;
  threadId: string;
  userId: string;
  userQuery: string;
  signal?: AbortSignal;
}): Promise<AgentJobCreateResponse> => {
  const response = await fetch(buildUrl(baseUrl, "/agent/api/jobs"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": userId,
    },
    body: JSON.stringify({
      thread_id: threadId,
      user_query: userQuery,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return (await response.json()) as AgentJobCreateResponse;
};

export const cancelAgentJob = async ({
  cancelUrl,
  userId,
}: {
  cancelUrl: string;
  userId: string;
}): Promise<void> => {
  const response = await fetch(cancelUrl, {
    method: "POST",
    headers: {
      "X-User-Id": userId,
    },
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
};

export const cancelAgentJobById = async ({
  baseUrl,
  jobId,
  userId,
}: {
  baseUrl: string;
  jobId: string;
  userId: string;
}): Promise<void> => {
  const response = await fetch(
    buildUrl(baseUrl, `/agent/api/jobs/${encodeURIComponent(jobId)}/cancel`),
    {
      method: "POST",
      headers: {
        "X-User-Id": userId,
      },
    },
  );

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }
};

export async function* streamAgentEvents({
  eventsUrl,
  userId,
  signal,
}: {
  eventsUrl: string;
  userId: string;
  signal?: AbortSignal;
}): AsyncGenerator<AgentEvent, void, unknown> {
  const response = await fetch(eventsUrl, {
    method: "GET",
    headers: {
      Accept: "text/event-stream",
      "X-User-Id": userId,
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  if (!response.body) {
    throw new Error("Agent event stream response body is empty");
  }

  for await (const sseEvent of parseSseStream(response.body)) {
    try {
      yield JSON.parse(sseEvent.data) as AgentEvent;
    } catch {
      // Ignore malformed events and continue reading the stream.
    }
  }
}
