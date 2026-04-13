export type SseEvent = {
  data: string;
  id?: string;
  event?: string;
};

const parseRawEvent = (rawEvent: string): SseEvent | null => {
  const lines = rawEvent.split(/\r?\n/);
  const data: string[] = [];
  let id: string | undefined;
  let event: string | undefined;

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }

    const separatorIndex = line.indexOf(":");
    const field =
      separatorIndex === -1 ? line : line.slice(0, separatorIndex).trim();
    const rawValue = separatorIndex === -1 ? "" : line.slice(separatorIndex + 1);
    const value = rawValue.startsWith(" ") ? rawValue.slice(1) : rawValue;

    if (field === "data") {
      data.push(value);
    } else if (field === "id") {
      id = value;
    } else if (field === "event") {
      event = value;
    }
  }

  if (data.length === 0) {
    return null;
  }

  return {
    data: data.join("\n"),
    event,
    id,
  };
};

export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const segments = buffer.split(/\r?\n\r?\n/);
      buffer = segments.pop() ?? "";

      for (const segment of segments) {
        const event = parseRawEvent(segment);
        if (event) {
          yield event;
        }
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      const event = parseRawEvent(buffer);
      if (event) {
        yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
