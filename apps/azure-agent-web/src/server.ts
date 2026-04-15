import type { FastifyInstance } from "fastify";

import { buildApp } from "./app.js";
import { config } from "./config.js";
import { createPostgresPool } from "./lib/db.js";
import { loadWebSecrets } from "./lib/secrets.js";
import { ThreadRepository } from "./lib/thread-repository.js";

const start = async () => {
  let app: FastifyInstance | null = null;

  try {
    const secrets = await loadWebSecrets({
      keyVaultUrl: config.keyVaultUrl,
    });

    const pool = secrets
      ? createPostgresPool({
          connectionString: secrets.postgresConnString,
        })
      : null;
    const threadRepository = pool ? new ThreadRepository(pool) : null;

    if (threadRepository) {
      await threadRepository.ensureSchema();
    }

    app = buildApp({
      threadRepository,
    });

    if (pool) {
      app.addHook("onClose", async () => {
        await pool.end();
      });
    }

    await app.listen({
      host: config.host,
      port: config.port,
    });
  } catch (error) {
    if (app) {
      app.log.error(error);
    } else {
      console.error(error);
    }
    process.exit(1);
  }
};

void start();
