import type {
  AttachmentAdapter,
  CompleteAttachment,
  PendingAttachment,
} from "@assistant-ui/react";

const MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024;

const inferAttachmentKind = (
  file: File,
): "image" | "document" | "file" => {
  if (file.type.startsWith("image/")) {
    return "image";
  }
  if (
    file.type === "application/pdf" ||
    file.type.startsWith("text/") ||
    file.type.includes("officedocument") ||
    file.type === "application/json"
  ) {
    return "document";
  }
  return "file";
};

const fileToDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsDataURL(file);
  });

export type AgentAttachmentAdapterOptions = {
  apiBaseUrl: string;
  userId: string;
  getThreadId: () => string | null | undefined;
};

export class AgentAttachmentAdapter implements AttachmentAdapter {
  accept = "*";

  constructor(private readonly options: AgentAttachmentAdapterOptions) {}

  async add({ file }: { file: File }): Promise<PendingAttachment> {
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      throw new Error(
        `File "${file.name}" exceeds the 25MB upload limit.`,
      );
    }

    return {
      id: crypto.randomUUID(),
      type: inferAttachmentKind(file),
      name: file.name,
      contentType: file.type || "application/octet-stream",
      file,
      status: { type: "requires-action", reason: "composer-send" },
    };
  }

  async send(attachment: PendingAttachment): Promise<CompleteAttachment> {
    const { apiBaseUrl, userId, getThreadId } = this.options;
    const threadId = getThreadId();
    if (!threadId) {
      throw new Error("Cannot upload attachment without an active thread.");
    }

    const form = new FormData();
    form.append("thread_id", threadId);
    form.append("file", attachment.file, attachment.name);

    const response = await fetch(`${apiBaseUrl}/api/files`, {
      method: "POST",
      headers: {
        "X-User-Id": userId,
      },
      body: form,
    });

    if (!response.ok) {
      let message = `Upload failed with status ${response.status}`;
      try {
        const payload = (await response.json()) as {
          detail?: string;
          error?: string;
        };
        message = payload.detail || payload.error || message;
      } catch {
        // ignore
      }
      throw new Error(message);
    }

    const result = (await response.json()) as {
      file_id: string;
      filename: string;
      mime_type: string | null;
      size: number;
    };

    const meta = {
      file_id: result.file_id,
      filename: result.filename,
      mime_type: result.mime_type,
      size: result.size,
    };

    const isImage = (attachment.contentType ?? "").startsWith("image/");
    const dataUrl = isImage
      ? await fileToDataUrl(attachment.file).catch(() => "")
      : "";

    const content = isImage && dataUrl
      ? [
          { type: "image" as const, image: dataUrl, filename: result.filename },
          { type: "data" as const, name: "agent-file", data: meta },
        ]
      : [{ type: "data" as const, name: "agent-file", data: meta }];

    return {
      id: attachment.id,
      type: attachment.type,
      name: result.filename,
      contentType: result.mime_type ?? attachment.contentType,
      content,
      status: { type: "complete" },
    };
  }

  async remove(): Promise<void> {
    // Uploads are referenced by file_id which the agent will discover via DB
    // when the next turn runs; deletion is handled by retention policy.
  }
}
