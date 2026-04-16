const toNumber = (value: string | undefined, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const splitCsv = (value: string | undefined): string[] => {
  if (!value) {
    return ["http://localhost:3000"];
  }

  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
};

const matchesOriginPattern = (origin: string, pattern: string) => {
  if (pattern === "*") {
    return true;
  }

  if (!pattern.includes("*")) {
    return origin === pattern;
  }

  try {
    const originUrl = new URL(origin);
    const patternUrl = new URL(pattern.replace("*.", "placeholder."));

    if (originUrl.protocol !== patternUrl.protocol) {
      return false;
    }

    const expectedSuffix = patternUrl.hostname.replace("placeholder.", ".");
    return originUrl.hostname.endsWith(expectedSuffix);
  } catch {
    return false;
  }
};

const corsOrigins = splitCsv(process.env.CORS_ORIGINS);

export const config = {
  host: process.env.HOST ?? "0.0.0.0",
  port: toNumber(process.env.PORT, 3001),
  corsOrigins,
  agentApiBaseUrl:
    process.env.AGENT_API_BASE_URL ?? "http://127.0.0.1:8080",
  defaultUserId: process.env.DEFAULT_USER_ID ?? "dev-user",
  keyVaultUrl: process.env.KEY_VAULT_URL?.trim(),
  isCorsOriginAllowed(origin: string | undefined) {
    if (!origin) {
      return false;
    }

    return corsOrigins.some((pattern) => matchesOriginPattern(origin, pattern));
  },
};
