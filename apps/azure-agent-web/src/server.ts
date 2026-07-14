import type { FastifyInstance } from "fastify";

import { buildApp } from "./app.js";
import { config } from "./config.js";
import { createPostgresPool } from "./lib/db.js";
import { ThreadRepository } from "./lib/thread-repository.js";

const formatStartupError = (error: unknown) => {
  if (error instanceof Error) {
    const details: Record<string, unknown> = {
      name: error.name,
      message: error.message,
    };

    if ("code" in error && typeof error.code === "string") {
      details.code = error.code;
    }

    if ("cause" in error && error.cause instanceof Error) {
      details.cause = {
        name: error.cause.name,
        message: error.cause.message,
        ...(typeof (error.cause as Error & { code?: unknown }).code === "string"
          ? { code: (error.cause as Error & { code?: string }).code }
          : {}),
      };
    }

    return details;
  }

  return {
    value: error,
  };
};

const start = async () => {
  let app: FastifyInstance | null = null;

  try {
    console.log("[startup] creating Postgres pool");
    const pool = createPostgresPool({
      connectionString: config.postgresWebConnString,
    });
    const threadRepository = new ThreadRepository(pool);

    console.log("[startup] ensuring Postgres schema");
    await threadRepository.ensureSchema();
    console.log("[startup] Postgres schema ready");

    console.log("[startup] building Fastify app");
    app = buildApp({
      threadRepository,
    });

    app.addHook("onClose", async () => {
      await pool.end();
    });

    app.log.info(
      {
        host: config.host,
        port: config.port,
      },
      "Starting web server",
    );

    await app.listen({
      host: config.host,
      port: config.port,
    });
  } catch (error) {
    const formattedError = formatStartupError(error);

    if (app) {
      app.log.error(
        {
          error: formattedError,
        },
        "Web server startup failed",
      );
    } else {
      console.error("[startup] failed", formattedError);
    }
    process.exit(1);
  }
};

void start();
