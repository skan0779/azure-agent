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

const splitOptionalCsv = (value: string | undefined): string[] => {
  if (!value) {
    return [];
  }

  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
};

const requiredEnv = (name: string): string => {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`${name} is required`);
  }
  if (
    value.length >= 2 &&
    value[0] === value[value.length - 1] &&
    (value[0] === "\"" || value[0] === "'")
  ) {
    return value.slice(1, -1);
  }
  return value;
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
const authAudiences = splitOptionalCsv(process.env.AZURE_AUTH_AUDIENCES);
const azureAuthTenantId = process.env.AZURE_AUTH_TENANT_ID?.trim();
const azureAuthApiClientId = process.env.AZURE_AUTH_API_CLIENT_ID?.trim();
const isAzureAuthConfigured = Boolean(azureAuthTenantId && azureAuthApiClientId);

export const config = {
  host: process.env.HOST ?? "0.0.0.0",
  port: toNumber(process.env.PORT, 3001),
  corsOrigins,
  agentApiBaseUrl:
    process.env.AGENT_API_BASE_URL ?? "http://127.0.0.1:8080",
  postgresWebConnString: requiredEnv("POSTGRES_WEB_CONN_STRING"),
  auth: {
    tenantId: azureAuthTenantId,
    apiClientId: azureAuthApiClientId,
    audiences: isAzureAuthConfigured
      ? [
          ...authAudiences,
          azureAuthApiClientId,
          `api://${azureAuthApiClientId}`,
        ].filter((value, index, values) => values.indexOf(value) === index)
      : authAudiences,
    requiredScope: process.env.AZURE_AUTH_REQUIRED_SCOPE?.trim() ?? "access_as_user",
  },
  isCorsOriginAllowed(origin: string | undefined) {
    if (!origin) {
      return false;
    }

    return corsOrigins.some((pattern) => matchesOriginPattern(origin, pattern));
  },
};
