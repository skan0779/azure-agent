import cors from "@fastify/cors";
import Fastify from "fastify";

import { config } from "./config.js";
import { buildChatRoutes } from "./routes/chat.js";
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
    origin: config.corsOrigins,
    credentials: true,
  });

  void app.register(healthRoutes);
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
