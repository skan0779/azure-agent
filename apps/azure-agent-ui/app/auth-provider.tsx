"use client";

import {
  InteractionRequiredAuthError,
  PublicClientApplication,
  type AccountInfo,
} from "@azure/msal-browser";
import { UserCircleIcon } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import type { AccessTokenProvider } from "@/lib/auth";

type AuthContextValue = {
  account: AccountInfo;
  getAccessToken: AccessTokenProvider;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const tenantId = process.env.NEXT_PUBLIC_AZURE_TENANT_ID?.trim();
const clientId = process.env.NEXT_PUBLIC_AZURE_CLIENT_ID?.trim();
const apiScope = process.env.NEXT_PUBLIC_AZURE_API_SCOPE?.trim();

const getAuthConfigError = () => {
  const missing = [
    !tenantId ? "NEXT_PUBLIC_AZURE_TENANT_ID" : null,
    !clientId ? "NEXT_PUBLIC_AZURE_CLIENT_ID" : null,
    !apiScope ? "NEXT_PUBLIC_AZURE_API_SCOPE" : null,
  ].filter(Boolean);

  return missing.length > 0
    ? `Missing auth configuration: ${missing.join(", ")}`
    : null;
};

let msalInstance: PublicClientApplication | null = null;

const getMsalInstance = () => {
  if (!tenantId || !clientId) {
    return null;
  }

  if (!msalInstance) {
    msalInstance = new PublicClientApplication({
      auth: {
        clientId,
        authority: `https://login.microsoftonline.com/${tenantId}`,
        redirectUri: window.location.origin,
        postLogoutRedirectUri: window.location.origin,
      },
      cache: {
        cacheLocation: "localStorage",
      },
    });
  }

  return msalInstance;
};

const LoadingScreen = () => (
  <div className="flex h-dvh items-center justify-center bg-[#212121] px-6 text-[#ececec]">
    <div className="text-sm text-[#9f9f9f]">Loading...</div>
  </div>
);

const AuthConfigErrorScreen = ({ message }: { message: string }) => (
  <div className="flex h-dvh items-center justify-center bg-background px-6">
    <div className="w-full max-w-xl rounded-lg border border-border bg-card p-6 text-card-foreground shadow-sm">
      <h1 className="text-lg font-semibold">Authentication is not configured</h1>
      <p className="mt-2 text-sm text-muted-foreground">{message}</p>
    </div>
  </div>
);

const SignInScreen = ({ onSignIn }: { onSignIn: () => void }) => (
  <div className="flex h-dvh items-center justify-center bg-[#212121] px-6 text-[#ececec]">
    <div className="w-full max-w-sm rounded-lg border border-white/10 bg-white/5 p-6 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-lg bg-white/10">
          <UserCircleIcon className="size-5" />
        </div>
        <div>
          <h1 className="text-base font-semibold">Azure Agent</h1>
          <p className="text-sm text-[#9f9f9f]">Sign in with Microsoft Entra ID</p>
        </div>
      </div>
      <Button
        type="button"
        className="mt-6 w-full"
        onClick={onSignIn}
      >
        Sign in
      </Button>
    </div>
  </div>
);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const configError = getAuthConfigError();
  const [isReady, setIsReady] = useState(false);
  const [account, setAccount] = useState<AccountInfo | null>(null);

  const instance = useMemo(() => {
    if (typeof window === "undefined") {
      return null;
    }

    return getMsalInstance();
  }, []);

  useEffect(() => {
    if (!instance || configError) {
      queueMicrotask(() => setIsReady(true));
      return;
    }

    let cancelled = false;

    void (async () => {
      await instance.initialize();
      const result = await instance.handleRedirectPromise();
      const activeAccount =
        result?.account ?? instance.getActiveAccount() ?? instance.getAllAccounts()[0] ?? null;

      if (activeAccount) {
        instance.setActiveAccount(activeAccount);
      }

      if (!cancelled) {
        setAccount(activeAccount);
        setIsReady(true);
      }
    })().catch(() => {
      if (!cancelled) {
        setIsReady(true);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [configError, instance]);

  const login = useCallback(() => {
    if (!instance || !apiScope) {
      return;
    }

    void instance.loginRedirect({
      scopes: [apiScope],
      prompt: "select_account",
    });
  }, [instance]);

  const logout = useCallback(async () => {
    if (!instance) {
      return;
    }

    await instance.logoutRedirect({
      account: instance.getActiveAccount() ?? account ?? undefined,
    });
  }, [account, instance]);

  const getAccessToken = useCallback(async () => {
    if (!instance || !apiScope) {
      throw new Error("Authentication is not configured");
    }

    const activeAccount = instance.getActiveAccount() ?? account;
    if (!activeAccount) {
      throw new Error("User is not signed in");
    }

    try {
      const result = await instance.acquireTokenSilent({
        account: activeAccount,
        scopes: [apiScope],
      });
      return result.accessToken;
    } catch (error) {
      if (error instanceof InteractionRequiredAuthError) {
        await instance.acquireTokenRedirect({
          account: activeAccount,
          scopes: [apiScope],
        });
      }
      throw error;
    }
  }, [account, instance]);

  const value = useMemo<AuthContextValue | null>(() => {
    if (!account) {
      return null;
    }

    return {
      account,
      getAccessToken,
      logout,
    };
  }, [account, getAccessToken, logout]);

  if (configError) {
    return <AuthConfigErrorScreen message={configError} />;
  }

  if (!isReady) {
    return <LoadingScreen />;
  }

  if (!account || !value) {
    return <SignInScreen onSignIn={login} />;
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside AuthProvider");
  }

  return value;
};
