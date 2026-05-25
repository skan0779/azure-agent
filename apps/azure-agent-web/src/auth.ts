import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { webcrypto } from "node:crypto";

import { config } from "./config.js";

type JwtHeader = {
  alg?: string;
  kid?: string;
  typ?: string;
};

type EntraAccessTokenClaims = {
  aud?: string;
  iss?: string;
  exp?: number;
  nbf?: number;
  oid?: string;
  tid?: string;
  sub?: string;
  scp?: string;
  roles?: string[];
  preferred_username?: string;
  name?: string;
};

type Jwk = JsonWebKey & {
  kid?: string;
  use?: string;
  alg?: string;
};

declare module "fastify" {
  interface FastifyRequest {
    auth?: {
      token: string;
      claims: EntraAccessTokenClaims;
      userId: string;
    };
  }
}

let jwksCache:
  | {
      expiresAt: number;
      keys: Jwk[];
    }
  | null = null;

const textEncoder = new TextEncoder();

const base64UrlDecode = (value: string): Uint8Array => {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(
    Math.ceil(value.length / 4) * 4,
    "=",
  );
  return Buffer.from(padded, "base64");
};

const parseJsonPart = <T>(value: string): T => {
  const decoded = Buffer.from(base64UrlDecode(value)).toString("utf8");
  return JSON.parse(decoded) as T;
};

const extractBearerToken = (authorization: unknown): string | null => {
  if (typeof authorization !== "string") {
    return null;
  }

  const [scheme, token] = authorization.split(" ");
  if (scheme?.toLowerCase() !== "bearer" || !token) {
    return null;
  }

  return token;
};

const getJwks = async (): Promise<Jwk[]> => {
  const now = Date.now();
  if (jwksCache && jwksCache.expiresAt > now) {
    return jwksCache.keys;
  }

  if (!config.auth.tenantId) {
    throw new Error("AZURE_AUTH_TENANT_ID is required");
  }

  const response = await fetch(
    `https://login.microsoftonline.com/${config.auth.tenantId}/discovery/v2.0/keys`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch Entra JWKS: ${response.status}`);
  }

  const payload = (await response.json()) as { keys?: Jwk[] };
  const keys = payload.keys ?? [];
  jwksCache = {
    keys,
    expiresAt: now + 60 * 60 * 1000,
  };
  return keys;
};

const verifyJwtSignature = async ({
  token,
  header,
}: {
  token: string;
  header: JwtHeader;
}) => {
  if (header.alg !== "RS256" || !header.kid) {
    return false;
  }

  const [encodedHeader, encodedPayload, encodedSignature] = token.split(".");
  if (!encodedHeader || !encodedPayload || !encodedSignature) {
    return false;
  }

  const keys = await getJwks();
  const jwk = keys.find((key) => key.kid === header.kid);
  if (!jwk) {
    jwksCache = null;
    const refreshedKeys = await getJwks();
    const refreshedJwk = refreshedKeys.find((key) => key.kid === header.kid);
    if (!refreshedJwk) {
      return false;
    }
    return verifyWithJwk({
      jwk: refreshedJwk,
      data: `${encodedHeader}.${encodedPayload}`,
      signature: encodedSignature,
    });
  }

  return verifyWithJwk({
    jwk,
    data: `${encodedHeader}.${encodedPayload}`,
    signature: encodedSignature,
  });
};

const verifyWithJwk = async ({
  jwk,
  data,
  signature,
}: {
  jwk: Jwk;
  data: string;
  signature: string;
}) => {
  const key = await webcrypto.subtle.importKey(
    "jwk",
    jwk,
    {
      name: "RSASSA-PKCS1-v1_5",
      hash: "SHA-256",
    },
    false,
    ["verify"],
  );

  return webcrypto.subtle.verify(
    "RSASSA-PKCS1-v1_5",
    key,
    base64UrlDecode(signature),
    textEncoder.encode(data),
  );
};

const hasRequiredScope = (claims: EntraAccessTokenClaims): boolean => {
  const requiredScope = config.auth.requiredScope;
  if (!requiredScope) {
    return true;
  }

  const scopes = new Set((claims.scp ?? "").split(" ").filter(Boolean));
  if (scopes.has(requiredScope)) {
    return true;
  }

  return Array.isArray(claims.roles) && claims.roles.includes(requiredScope);
};

const validateClaims = (claims: EntraAccessTokenClaims): string | null => {
  const nowSeconds = Math.floor(Date.now() / 1000);
  const expectedIssuers = new Set([
    `https://login.microsoftonline.com/${config.auth.tenantId}/v2.0`,
    `https://sts.windows.net/${config.auth.tenantId}/`,
  ]);

  if (!claims.iss || !expectedIssuers.has(claims.iss)) {
    return "invalid_issuer";
  }

  if (!claims.aud || !config.auth.audiences.includes(claims.aud)) {
    return "invalid_audience";
  }

  if (!claims.exp || claims.exp <= nowSeconds) {
    return "token_expired";
  }

  if (claims.nbf && claims.nbf > nowSeconds) {
    return "token_not_yet_valid";
  }

  if (!hasRequiredScope(claims)) {
    return "insufficient_scope";
  }

  return null;
};

const resolveUserIdFromClaims = (claims: EntraAccessTokenClaims): string | null => {
  if (claims.tid && claims.oid) {
    return `${claims.tid}:${claims.oid}`;
  }

  return claims.oid ?? claims.sub ?? null;
};

export const verifyEntraAccessToken = async (token: string) => {
  const [encodedHeader, encodedPayload, encodedSignature] = token.split(".");
  if (!encodedHeader || !encodedPayload || !encodedSignature) {
    throw new Error("malformed_token");
  }

  const header = parseJsonPart<JwtHeader>(encodedHeader);
  const claims = parseJsonPart<EntraAccessTokenClaims>(encodedPayload);
  const isValidSignature = await verifyJwtSignature({ token, header });
  if (!isValidSignature) {
    throw new Error("invalid_signature");
  }

  const claimError = validateClaims(claims);
  if (claimError) {
    throw new Error(claimError);
  }

  const userId = resolveUserIdFromClaims(claims);
  if (!userId) {
    throw new Error("missing_user_claim");
  }

  return {
    claims,
    userId,
  };
};

const sendUnauthorized = (
  reply: FastifyReply,
  detail = "Bearer access token is required",
) => {
  reply.code(401);
  reply.header("WWW-Authenticate", "Bearer");
  return {
    error: "unauthorized",
    detail,
  };
};

export const registerAuth = (app: FastifyInstance) => {
  if (!config.auth.tenantId || !config.auth.apiClientId) {
    throw new Error(
      "AZURE_AUTH_TENANT_ID and AZURE_AUTH_API_CLIENT_ID are required",
    );
  }

  app.addHook("preHandler", async (request, reply) => {
    if (request.method === "OPTIONS" || request.url === "/health") {
      return;
    }

    const token = extractBearerToken(request.headers.authorization);
    if (!token) {
      return reply.send(sendUnauthorized(reply));
    }

    try {
      const result = await verifyEntraAccessToken(token);
      request.auth = {
        token,
        claims: result.claims,
        userId: result.userId,
      };
    } catch (error) {
      request.log.warn({ err: error }, "Rejected access token");
      return reply.send(sendUnauthorized(reply, "Invalid or expired access token"));
    }
  });
};

export const getRequestUserId = (request: FastifyRequest): string | null => {
  return request.auth?.userId ?? null;
};
