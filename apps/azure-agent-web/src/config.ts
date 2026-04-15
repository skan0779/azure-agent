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

export const config = {
  host: process.env.HOST ?? "0.0.0.0",
  port: toNumber(process.env.PORT, 3001),
  corsOrigins: splitCsv(process.env.CORS_ORIGINS),
  agentApiBaseUrl:
    process.env.AGENT_API_BASE_URL ?? "http://127.0.0.1:8080",
  defaultUserId: process.env.DEFAULT_USER_ID ?? "dev-user",
  keyVaultUrl: process.env.KEY_VAULT_URL?.trim(),
};
