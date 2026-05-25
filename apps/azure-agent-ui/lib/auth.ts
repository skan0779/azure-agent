export type AccessTokenProvider = () => Promise<string>;

export const buildBearerHeaders = async (
  getAccessToken: AccessTokenProvider,
  headers?: HeadersInit,
): Promise<HeadersInit> => {
  const token = await getAccessToken();
  return {
    ...headers,
    Authorization: `Bearer ${token}`,
  };
};
