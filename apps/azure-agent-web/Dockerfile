FROM node:20-bookworm-slim AS base

ENV PNPM_HOME="/pnpm" \
    PATH="/pnpm:$PATH"

RUN corepack enable

WORKDIR /app

FROM base AS deps

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/azure-agent-web/package.json ./apps/azure-agent-web/package.json

RUN pnpm install --frozen-lockfile --filter azure-agent-web...

FROM deps AS build

COPY apps/azure-agent-web/tsconfig.json ./apps/azure-agent-web/tsconfig.json
COPY apps/azure-agent-web/src ./apps/azure-agent-web/src

RUN pnpm --dir apps/azure-agent-web build

FROM base AS prod-deps

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/azure-agent-web/package.json ./apps/azure-agent-web/package.json

RUN pnpm install --frozen-lockfile --prod --filter azure-agent-web...

FROM node:20-bookworm-slim AS runtime

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3001

WORKDIR /app/apps/azure-agent-web

RUN useradd -m -u 10001 appuser

COPY --from=prod-deps /app/node_modules /app/node_modules
COPY --from=prod-deps /app/apps/azure-agent-web/package.json ./package.json
COPY --from=prod-deps /app/apps/azure-agent-web/node_modules ./node_modules
COPY --from=build /app/apps/azure-agent-web/dist ./dist

USER appuser

EXPOSE 3001

CMD ["node", "dist/server.js"]
