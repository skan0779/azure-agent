import cors from "@fastify/cors";
import Fastify from "fastify";

import { config } from "./config.js";
import { chatRoutes } from "./routes/chat.js";
import { healthRoutes } from "./routes/health.js";

export const buildApp = () => {
  const app = Fastify({
    logger: true,
  });

  void app.register(cors, {
    origin: config.corsOrigins,
    credentials: true,
  });

  void app.register(healthRoutes);
  void app.register(chatRoutes);

  return app;
};
