"use client";

import {
  ActionBarPrimitive,
  AuiIf,
  AttachmentPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import type { FC } from "react";
import { useEffect, useMemo } from "react";
import { useShallow } from "zustand/shallow";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  ExternalLinkIcon,
  FileTextIcon,
  LightbulbIcon,
  LoaderIcon,
  PencilIcon,
  PenLineIcon,
  PlusIcon,
  RotateCcwIcon,
  SquareIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
  XIcon,
} from "lucide-react";

import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { ToolFallback } from "@/components/assistant-ui/tool-fallback";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  CitationList,
  type SerializableCitation,
} from "@/components/tool-ui/citation/index";

const toSerializableCitation = (part: unknown): SerializableCitation | null => {
  if (!part || typeof part !== "object") {
    return null;
  }

  const candidate = part as Record<string, unknown>;

  if (candidate.type === "data" && candidate.name === "citation") {
    return toSerializableCitation(candidate.data);
  }

  if (
    !("href" in candidate) &&
    !("citationId" in candidate) &&
    !("title" in candidate)
  ) {
    return null;
  }

  if (typeof candidate.href !== "string" || candidate.href.length === 0) {
    return null;
  }

  return {
    id:
      typeof candidate.citationId === "string" && candidate.citationId.length > 0
        ? candidate.citationId
        : typeof candidate.id === "string" && candidate.id.length > 0
          ? candidate.id
          : candidate.href,
    href: candidate.href,
    title:
      typeof candidate.title === "string" && candidate.title.length > 0
        ? candidate.title
        : "Untitled source",
    snippet:
      typeof candidate.snippet === "string" ? candidate.snippet : undefined,
    domain: typeof candidate.domain === "string" ? candidate.domain : undefined,
    favicon:
      typeof candidate.favicon === "string" ? candidate.favicon : undefined,
    author: typeof candidate.author === "string" ? candidate.author : undefined,
    publishedAt:
      typeof candidate.publishedAt === "string"
        ? candidate.publishedAt
        : undefined,
    type:
      typeof candidate.citationType === "string"
        ? (candidate.citationType as SerializableCitation["type"])
        : undefined,
  };
};

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root className="flex h-full flex-col items-stretch bg-background px-4 text-foreground dark:bg-[#212121] dark:text-foreground">
      <AuiIf condition={(s) => s.thread.isEmpty}>
        <EmptyThreadView />
      </AuiIf>
      <AuiIf condition={(s) => !s.thread.isEmpty}>
        <ConversationThreadView />
      </AuiIf>
    </ThreadPrimitive.Root>
  );
};

const EmptyThreadView: FC = () => {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mx-auto flex h-full w-full max-w-5xl flex-1 items-center justify-center px-4 pb-8 pt-12">
        <div className="w-full max-w-4xl">
          <div className="mx-auto max-w-3xl">
            <div className="mb-10 flex flex-col items-start text-left">
              <p className="text-3xl font-medium tracking-tight text-white">
                Hello there!
              </p>
              <p className="mt-3 text-3xl tracking-tight text-[#9f9f9f]">
                How can I help you today?
              </p>
            </div>
            <EmptyComposer />
          </div>

          <div className="mx-auto mt-6 grid max-w-3xl gap-3 sm:grid-cols-2">
            <EmptySuggestionChip
              icon={<LightbulbIcon className="size-4" />}
              label="Help me learn"
            />
            <EmptySuggestionChip
              icon={<PenLineIcon className="size-4" />}
              label="Write anything"
            />
          </div>

          <p className="mt-6 text-center text-muted-foreground text-xs dark:text-[#8f8f8f]">
            Agent can make mistakes. Check important info.
          </p>
        </div>
      </div>
    </div>
  );
};

const EmptySuggestionChip = ({
  icon,
  label,
}: {
  icon: React.ReactNode;
  label: string;
}) => {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-[#d6d6d6] shadow-sm transition hover:bg-white/8">
      <div className="text-[#bdbdbd]">{icon}</div>
      <span>{label}</span>
    </div>
  );
};

const ConversationThreadView: FC = () => {
  return (
    <ThreadPrimitive.Viewport className="aui-scrollbar flex grow flex-col gap-8 overflow-y-auto pt-16">
      <ThreadPrimitive.Messages>
        {() => <ThreadMessage />}
      </ThreadPrimitive.Messages>

      <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mx-auto mt-auto flex w-full max-w-3xl flex-col gap-4 overflow-visible rounded-t-3xl bg-background pb-2 dark:bg-[#212121]">
        <ThreadScrollToBottom />
        <Composer />
        <p className="text-center text-muted-foreground text-xs dark:text-[#cdcdcd]">
          Agent can make mistakes. Check important info.
        </p>
      </ThreadPrimitive.ViewportFooter>
    </ThreadPrimitive.Viewport>
  );
};

const ThreadMessage: FC = () => {
  const isEditing = useAuiState((s) => s.message.composer.isEditing);
  const role = useAuiState((s) => s.message.role);

  if (isEditing) {
    return <EditComposer />;
  }

  if (role === "user") {
    return <UserMessage />;
  }

  return <AssistantMessage />;
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="w-full rounded-3xl border pl-2 dark:border-none dark:bg-white/5">
      <AuiIf condition={(s) => s.composer.attachments.length > 0}>
        <div className="flex flex-row flex-wrap gap-2 px-1 py-3">
          <ComposerPrimitive.Attachments components={{ Attachment: ChatGPTAttachmentUI }} />
        </div>
      </AuiIf>

      <div className="flex items-center justify-center">
        <ComposerPrimitive.AddAttachment className="flex size-8 items-center justify-center overflow-hidden rounded-full hover:bg-foreground/5 dark:hover:bg-foreground/15">
          <PlusIcon size={18} />
        </ComposerPrimitive.AddAttachment>

        <ComposerPrimitive.Input
          placeholder="Ask anything"
          className="h-12 max-h-40 grow resize-none bg-transparent p-3.5 text-foreground text-sm outline-none placeholder:text-muted-foreground dark:text-white dark:placeholder:text-white/50"
        />

        <AuiIf condition={(s) => !s.thread.isRunning}>
          <ComposerPrimitive.Send className="m-2 flex size-8 items-center justify-center rounded-full bg-primary text-primary-foreground transition-opacity disabled:opacity-10 dark:bg-white dark:text-black">
            <ArrowUpIcon className="size-5" />
          </ComposerPrimitive.Send>
        </AuiIf>

        <AuiIf condition={(s) => s.thread.isRunning}>
          <ComposerPrimitive.Cancel className="m-2 flex size-8 items-center justify-center rounded-full bg-primary text-primary-foreground dark:bg-white">
            <SquareIcon className="size-3 fill-current dark:text-black" />
          </ComposerPrimitive.Cancel>
        </AuiIf>
      </div>
    </ComposerPrimitive.Root>
  );
};

const EmptyComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="w-full rounded-[2rem] border border-white/10 bg-white/5 p-3 shadow-[0_12px_40px_-24px_rgba(0,0,0,0.8)] backdrop-blur-sm">
      <AuiIf condition={(s) => s.composer.attachments.length > 0}>
        <div className="mb-2 flex flex-row flex-wrap gap-2">
          <ComposerPrimitive.Attachments
            components={{ Attachment: ChatGPTAttachmentUI }}
          />
        </div>
      </AuiIf>

      <ComposerPrimitive.Input
        placeholder="Ask anything"
        className="min-h-16 w-full resize-none bg-transparent px-3 py-2 text-[15px] text-foreground outline-none placeholder:text-muted-foreground dark:text-white dark:placeholder:text-white/50"
      />

      <div className="mt-2 flex items-center justify-between gap-3 px-1">
        <div className="flex items-center">
          <ComposerPrimitive.AddAttachment className="flex size-9 items-center justify-center rounded-full text-[#cfcfcf] transition hover:bg-white/10 hover:text-white">
            <PlusIcon className="size-4" />
          </ComposerPrimitive.AddAttachment>
        </div>

        <div className="flex items-center gap-2">
          <AuiIf condition={(s) => !s.thread.isRunning}>
            <ComposerPrimitive.Send className="flex size-9 items-center justify-center rounded-full bg-white text-black transition-opacity disabled:opacity-30">
              <ArrowUpIcon className="size-4" />
            </ComposerPrimitive.Send>
          </AuiIf>

          <AuiIf condition={(s) => s.thread.isRunning}>
            <ComposerPrimitive.Cancel className="flex size-9 items-center justify-center rounded-full bg-white text-black">
              <SquareIcon className="size-3 fill-current" />
            </ComposerPrimitive.Cancel>
          </AuiIf>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        className="absolute -top-10 z-10 self-center rounded-full border bg-background p-2 shadow-sm disabled:invisible dark:border-white/15 dark:bg-[#2a2a2a]"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative mx-auto flex w-full max-w-3xl flex-col items-end gap-1">
      <div className="flex flex-row flex-wrap justify-end gap-2">
        <MessagePrimitive.Attachments components={{ Attachment: ChatGPTAttachmentUI }} />
      </div>

      <div className="flex items-start gap-4">
        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="not-last"
          autohideFloat="single-branch"
          className="mt-2 flex items-center gap-1"
        >
          <ActionBarPrimitive.Copy asChild>
            <TooltipIconButton tooltip="Copy" className="text-[#b4b4b4]">
              <AuiIf condition={(s) => s.message.isCopied}>
                <CheckIcon />
              </AuiIf>
              <AuiIf condition={(s) => !s.message.isCopied}>
                <CopyIcon />
              </AuiIf>
            </TooltipIconButton>
          </ActionBarPrimitive.Copy>
          <AuiIf
            condition={(s) =>
              s.message.isLast ||
              (s.message.index === s.thread.messages.length - 2 &&
                s.thread.messages.at(-1)?.role === "assistant")
            }
          >
            <ActionBarPrimitive.Edit asChild>
              <TooltipIconButton tooltip="Edit" className="text-[#b4b4b4]">
                <PencilIcon />
              </TooltipIconButton>
            </ActionBarPrimitive.Edit>
          </AuiIf>
        </ActionBarPrimitive.Root>

        <div className="rounded-3xl bg-secondary px-5 py-2 text-foreground dark:bg-white/5 dark:text-[#eee]">
          <MessagePrimitive.Parts />
        </div>
      </div>

      <MessageBranchPicker align="end" />
    </MessagePrimitive.Root>
  );
};

const EditComposer: FC = () => {
  return (
    <ComposerPrimitive.Root className="mx-auto flex w-full max-w-3xl flex-col justify-end gap-1 rounded-3xl bg-secondary dark:bg-white/15">
      <ComposerPrimitive.Input className="flex h-8 w-full resize-none bg-transparent p-5 pb-0 text-foreground outline-none dark:text-white" />

      <div className="m-3 mt-2 flex items-center justify-center gap-2 self-end">
        <ComposerPrimitive.Cancel className="rounded-full bg-background px-3 py-2 font-semibold text-foreground text-sm hover:bg-muted dark:bg-zinc-900 dark:text-white dark:hover:bg-zinc-800">
          Cancel
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send className="rounded-full bg-primary px-3 py-2 font-semibold text-primary-foreground text-sm hover:bg-primary/90 dark:bg-white dark:text-black dark:hover:bg-white/90">
          Send
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  const messageContent = useAuiState((s) => s.message.content);
  const citations = useMemo(() => {
    return messageContent.flatMap((part): SerializableCitation[] => {
      const citation = toSerializableCitation(part);
      return citation ? [citation] : [];
    });
  }, [messageContent]);

  return (
    <MessagePrimitive.Root className="relative mx-auto flex w-full max-w-3xl">
      <div className="min-w-0 flex-1 pt-1">
        <div className="text-foreground dark:text-[#eee]">
          <MessagePrimitive.Parts>
            {({ part }) => {
              if (part.type === "text") return <MarkdownText />;
              if (part.type === "file") return <AssistantFilePart part={part} />;
              if (part.type === "tool-call")
                return part.toolUI ?? <ToolFallback {...part} />;
              return null;
            }}
          </MessagePrimitive.Parts>
          <AuiIf
            condition={(s) => {
              if (!s.thread.isRunning) {
                return false;
              }

              return !s.message.content.some((part) => {
                if (!part || typeof part !== "object" || !("type" in part)) {
                  return false;
                }

                if (part.type === "text") {
                  return "text" in part && typeof part.text === "string"
                    ? part.text.trim().length > 0
                    : false;
                }

                return part.type === "tool-call" || part.type === "file";
              });
            }}
          >
            <div className="mt-2 flex items-center gap-2 text-[#9f9f9f]">
              <LoaderIcon className="size-4 animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          </AuiIf>
          <MessageError />
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-2">
          <ActionBarPrimitive.Root
            hideWhenRunning
            className="flex items-center gap-1 rounded-lg"
          >
            <ActionBarPrimitive.FeedbackPositive asChild>
              <TooltipIconButton tooltip="Good response" className="text-[#b4b4b4]">
                <ThumbsUpIcon />
              </TooltipIconButton>
            </ActionBarPrimitive.FeedbackPositive>
            <ActionBarPrimitive.FeedbackNegative asChild>
              <TooltipIconButton tooltip="Bad response" className="text-[#b4b4b4]">
                <ThumbsDownIcon />
              </TooltipIconButton>
            </ActionBarPrimitive.FeedbackNegative>
            <AuiIf condition={(s) => s.message.isLast}>
              <ActionBarPrimitive.Reload asChild>
                <TooltipIconButton tooltip="Reload" className="text-[#b4b4b4]">
                  <RotateCcwIcon />
                </TooltipIconButton>
              </ActionBarPrimitive.Reload>
            </AuiIf>
            <ActionBarPrimitive.Copy asChild>
              <TooltipIconButton tooltip="Copy" className="text-[#b4b4b4]">
                <AuiIf condition={(s) => s.message.isCopied}>
                  <CheckIcon />
                </AuiIf>
                <AuiIf condition={(s) => !s.message.isCopied}>
                  <CopyIcon />
                </AuiIf>
              </TooltipIconButton>
            </ActionBarPrimitive.Copy>
          </ActionBarPrimitive.Root>
          {citations.length > 0 ? (
            <CitationList
              id={`message-citations-${citations[0]?.id ?? "unknown"}`}
              citations={citations}
              variant="stacked"
            />
          ) : null}
        </div>

        <MessageBranchPicker align="start" />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantFilePart = ({
  part,
}: {
  part: {
    filename?: string;
    mimeType: string;
    data: string;
  };
}) => {
  const fileLabel = part.filename?.trim() || "Attached file";

  return (
    <a
      href={part.data}
      target="_blank"
      rel="noreferrer"
      className="mt-2 flex w-full max-w-md items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left transition hover:bg-white/10"
    >
      <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-white/10 text-[#d8d8d8]">
        <FileTextIcon className="size-5" />
      </div>

      <div className="min-w-0 flex-1">
        <div className="truncate font-medium text-sm text-[#f1f1f1]">
          {fileLabel}
        </div>
        <div className="truncate text-xs text-[#9f9f9f]">{part.mimeType}</div>
      </div>

      <ExternalLinkIcon className="size-4 shrink-0 text-[#8f8f8f]" />
    </a>
  );
};

const MessageBranchPicker = ({
  align,
}: {
  align: "start" | "end";
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={`flex items-center gap-1 pt-1 ${align === "end" ? "self-end" : "self-start"}`}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous branch" className="text-[#8f8f8f]">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <div className="min-w-10 text-center text-[#8f8f8f] text-xs tabular-nums">
        <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
      </div>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next branch" className="text-[#8f8f8f]">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="mt-2 rounded-md border border-destructive bg-destructive/10 p-3 text-destructive text-sm dark:bg-destructive/5 dark:text-red-200">
        <ErrorPrimitive.Message className="line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const useFileSrc = (file: File | undefined) => {
  const src = useMemo(() => {
    if (!file) {
      return undefined;
    }

    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (src) {
        URL.revokeObjectURL(src);
      }
    };
  }, [src]);

  return src;
};

const useAttachmentSrc = () => {
  const { file, src } = useAuiState(
    useShallow((s): { file?: File; src?: string } => {
      if (s.attachment.type !== "image") return {};
      if (s.attachment.file) return { file: s.attachment.file };
      const src = s.attachment.content?.filter((c) => c.type === "image")[0]?.image;
      if (!src) return {};
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

const ChatGPTAttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source !== "message";
  const src = useAttachmentSrc();

  return (
    <AttachmentPrimitive.Root className="group/attachment relative">
      <div className="flex items-center gap-2 overflow-hidden rounded-2xl border bg-secondary dark:bg-white/5">
        <AuiIf condition={(s) => s.attachment.type === "image"}>
          {src ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img className="size-32 rounded-md object-cover" alt="Attachment" src={src} />
          ) : (
            <div className="flex h-full w-12 items-center justify-center rounded-md">
              <AttachmentPrimitive.unstable_Thumb className="text-xs" />
            </div>
          )}
        </AuiIf>
        <AuiIf condition={(s) => s.attachment.type !== "image"}>
          <div className="flex h-full w-12 items-center justify-center rounded-[9px] bg-background text-[#6b6b6b] dark:bg-[#3a3a3a] dark:text-[#9a9a9a]">
            <AttachmentPrimitive.unstable_Thumb className="text-xs" />
          </div>
        </AuiIf>
      </div>
      {isComposer && (
        <AttachmentPrimitive.Remove className="absolute -top-1.5 -right-1.5 flex size-7 items-center justify-center rounded-full border border-[#e5e5e5] bg-white text-[#6b6b6b] transition-all hover:bg-[#f5f5f5] hover:text-[#0d0d0d] dark:border-[#3a3a3a] dark:bg-[#1a1a1a] dark:text-[#9a9a9a] dark:hover:bg-[#252525] dark:hover:text-white">
          <XIcon size={14} />
        </AttachmentPrimitive.Remove>
      )}
    </AttachmentPrimitive.Root>
  );
};
