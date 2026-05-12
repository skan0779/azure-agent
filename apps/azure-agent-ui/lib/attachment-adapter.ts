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
  // Resolves the thread UUID, creating one on demand when the thread
  // has not been initialized yet (i.e. first attachment in a new chat).
  resolveThreadId: () => Promise<string>;
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
    const { apiBaseUrl, userId, resolveThreadId } = this.options;
    const threadId = await resolveThreadId();

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

    const mimeType = result.mime_type ?? "application/octet-stream";
    const isImage = mimeType.startsWith("image/");
    const dataUrl = isImage
      ? await fileToDataUrl(attachment.file).catch(() => "")
      : "";

    // Backend resolves file_id from this URL path (see chat.ts).
    const downloadUrl = `${apiBaseUrl}/api/files/${encodeURIComponent(result.file_id)}/download`;

    const content = isImage && dataUrl
      ? [
          { type: "image" as const, image: dataUrl, filename: result.filename },
        ]
      : [
          {
            type: "file" as const,
            filename: result.filename,
            mimeType,
            data: downloadUrl,
          },
        ];

    return {
      id: attachment.id,
      type: attachment.type,
      name: result.filename,
      contentType: mimeType,
      content,
      status: { type: "complete" },
    };
  }

  async remove(): Promise<void> {
    // Uploads are referenced by file_id which the agent will discover via DB
    // when the next turn runs; deletion is handled by retention policy.
  }
}
