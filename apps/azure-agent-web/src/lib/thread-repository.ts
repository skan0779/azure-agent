import { listThreadMessagesResponseSchema, listThreadsResponseSchema } from "./thread-history.js";
import type {
  ListThreadMessagesResponse,
  ListThreadsResponse,
  ThreadSummary,
  UIMessage,
} from "./thread-history.js";
import type { PostgresPool } from "./db.js";

const THREADS_TABLE_SQL = `
  CREATE TABLE IF NOT EXISTS agent_threads (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_job_id TEXT NULL,
    title_source TEXT NULL
  );
`;

const THREAD_MESSAGES_TABLE_SQL = `
  CREATE TABLE IF NOT EXISTS agent_thread_messages (
    id TEXT PRIMARY KEY,
    thread_id UUID NOT NULL REFERENCES agent_threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    metadata JSONB NULL,
    parts JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );
`;

const THREADS_INDEX_SQL = `
  CREATE INDEX IF NOT EXISTS agent_threads_user_updated_idx
  ON agent_threads (user_id, updated_at DESC);
`;

const THREAD_MESSAGES_INDEX_SQL = `
  CREATE INDEX IF NOT EXISTS agent_thread_messages_thread_created_idx
  ON agent_thread_messages (thread_id, created_at ASC);
`;

export class ThreadRepository {
  constructor(private readonly pool: PostgresPool) {}

  async ensureSchema() {
    await this.pool.query(THREADS_TABLE_SQL);
    await this.pool.query(THREAD_MESSAGES_TABLE_SQL);
    await this.pool.query(THREADS_INDEX_SQL);
    await this.pool.query(THREAD_MESSAGES_INDEX_SQL);
  }

  async listThreadsForUser(userId: string): Promise<ListThreadsResponse> {
    const result = await this.pool.query<{
      id: string;
      title: string;
      created_at: Date;
      updated_at: Date;
      last_job_id: string | null;
      title_source: "manual" | "first-user-message" | "generated" | null;
    }>(
      `
        SELECT
          id::text AS id,
          title,
          created_at,
          updated_at,
          last_job_id,
          title_source
        FROM agent_threads
        WHERE user_id = $1
        ORDER BY updated_at DESC, created_at DESC
      `,
      [userId],
    );

    return listThreadsResponseSchema.parse(
      result.rows.map((row) => ({
        id: row.id,
        title: row.title,
        createdAt: row.created_at.toISOString(),
        updatedAt: row.updated_at.toISOString(),
        lastJobId: row.last_job_id ?? undefined,
        titleSource: row.title_source ?? undefined,
      })),
    );
  }

  async getThreadMessages({
    threadId,
    userId,
  }: {
    threadId: string;
    userId: string;
  }): Promise<ListThreadMessagesResponse> {
    const exists = await this.pool.query<{ id: string }>(
      `
        SELECT id::text AS id
        FROM agent_threads
        WHERE id = $1::uuid AND user_id = $2
      `,
      [threadId, userId],
    );

    if (exists.rowCount === 0) {
      return [];
    }

    const result = await this.pool.query<{
      id: string;
      role: "system" | "user" | "assistant";
      metadata: unknown;
      parts: unknown;
    }>(
      `
        SELECT
          id,
          role,
          metadata,
          parts
        FROM agent_thread_messages
        WHERE thread_id = $1::uuid
        ORDER BY created_at ASC, id ASC
      `,
      [threadId],
    );

    return listThreadMessagesResponseSchema.parse(
      result.rows.map((row) => ({
        id: row.id,
        role: row.role,
        metadata: row.metadata ?? undefined,
        parts: row.parts,
      })),
    );
  }

  async upsertThread({
    threadId,
    userId,
    title,
    updatedAt,
    lastJobId,
    titleSource,
  }: {
    threadId: string;
    userId: string;
    title: string;
    updatedAt?: string;
    lastJobId?: string;
    titleSource?: "manual" | "first-user-message" | "generated";
  }): Promise<ThreadSummary> {
    const updatedAtValue = updatedAt ?? new Date().toISOString();

    const result = await this.pool.query<{
      id: string;
      title: string;
      created_at: Date;
      updated_at: Date;
      last_job_id: string | null;
      title_source: "manual" | "first-user-message" | "generated" | null;
    }>(
      `
        INSERT INTO agent_threads (
          id,
          user_id,
          title,
          created_at,
          updated_at,
          last_job_id,
          title_source
        )
        VALUES (
          $1::uuid,
          $2,
          $3,
          $4::timestamptz,
          $4::timestamptz,
          $5,
          $6
        )
        ON CONFLICT (id)
        DO UPDATE SET
          updated_at = EXCLUDED.updated_at,
          last_job_id = COALESCE(EXCLUDED.last_job_id, agent_threads.last_job_id),
          title = CASE
            WHEN agent_threads.title_source = 'manual' THEN agent_threads.title
            ELSE EXCLUDED.title
          END,
          title_source = CASE
            WHEN agent_threads.title_source = 'manual' THEN agent_threads.title_source
            ELSE EXCLUDED.title_source
          END
        WHERE agent_threads.user_id = EXCLUDED.user_id
        RETURNING
          id::text AS id,
          title,
          created_at,
          updated_at,
          last_job_id,
          title_source
      `,
      [
        threadId,
        userId,
        title,
        updatedAtValue,
        lastJobId ?? null,
        titleSource ?? null,
      ],
    );

    return listThreadsResponseSchema.element.parse({
      id: result.rows[0].id,
      title: result.rows[0].title,
      createdAt: result.rows[0].created_at.toISOString(),
      updatedAt: result.rows[0].updated_at.toISOString(),
      lastJobId: result.rows[0].last_job_id ?? undefined,
      titleSource: result.rows[0].title_source ?? undefined,
    });
  }

  async upsertMessage({
    threadId,
    message,
  }: {
    threadId: string;
    message: UIMessage;
  }) {
    await this.pool.query(
      `
        INSERT INTO agent_thread_messages (
          id,
          thread_id,
          role,
          metadata,
          parts
        )
        VALUES (
          $1,
          $2::uuid,
          $3,
          $4::jsonb,
          $5::jsonb
        )
        ON CONFLICT (id)
        DO UPDATE SET
          role = EXCLUDED.role,
          metadata = EXCLUDED.metadata,
          parts = EXCLUDED.parts
      `,
      [
        message.id,
        threadId,
        message.role,
        JSON.stringify(message.metadata ?? null),
        JSON.stringify(message.parts),
      ],
    );
  }

  async deleteThread({
    threadId,
    userId,
  }: {
    threadId: string;
    userId: string;
  }): Promise<void> {
    await this.pool.query(
      `
        DELETE FROM agent_threads
        WHERE id = $1::uuid AND user_id = $2
      `,
      [threadId, userId],
    );
  }
}
