import cors from "@fastify/cors";
import Fastify from "fastify";

import { config } from "./config.js";
import { buildChatRoutes } from "./routes/chat.js";
import { filesRoutes } from "./routes/files.js";
import { healthRoutes } from "./routes/health.js";
import { buildThreadsRoutes } from "./routes/threads.js";
import type { ThreadRepository } from "./lib/thread-repository.js";

export const buildApp = ({
  threadRepository = null,
}: {
  threadRepository?: ThreadRepository | null;
} = {}) => {
  const app = Fastify({
    logger: true,
  });

  void app.register(cors, {
    origin(origin, callback) {
      if (!origin || config.isCorsOriginAllowed(origin)) {
        callback(null, true);
        return;
      }

      callback(null, false);
    },
    credentials: true,
    allowedHeaders: ["Content-Type", "X-User-Id"],
    methods: ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
  });

  void app.register(healthRoutes);
  void app.register(filesRoutes);
  void app.register(
    buildChatRoutes({
      threadRepository,
    }),
  );
  void app.register(
    buildThreadsRoutes({
      threadRepository,
    }),
  );

  return app;
};
