import { DefaultAzureCredential } from "@azure/identity";
import { SecretClient } from "@azure/keyvault-secrets";

export type WebSecrets = {
  postgresConnString: string;
};

const POSTGRES_CONN_STRING_SECRET_NAME = "POSTGRES-WEB-CONN-STRING";

export const loadWebSecrets = async ({
  keyVaultUrl,
}: {
  keyVaultUrl?: string;
}): Promise<WebSecrets | null> => {
  if (!keyVaultUrl) {
    return null;
  }

  const credential = new DefaultAzureCredential();
  const client = new SecretClient(keyVaultUrl, credential);
  const postgresConnString = await client.getSecret(
    POSTGRES_CONN_STRING_SECRET_NAME,
  );

  if (!postgresConnString.value) {
    throw new Error(
      `Missing Key Vault secret value: ${POSTGRES_CONN_STRING_SECRET_NAME}`,
    );
  }

  return {
    postgresConnString: postgresConnString.value,
  };
};
