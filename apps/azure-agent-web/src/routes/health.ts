import type { FastifyPluginAsync } from "fastify";

export const healthRoutes: FastifyPluginAsync = async (app) => {
  app.get("/health", async () => {
    return {
      ok: true,
      service: "azure-agent-web",
    };
  });
};
