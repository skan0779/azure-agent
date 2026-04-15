import { Pool } from "pg";

export const createPostgresPool = ({
  connectionString,
}: {
  connectionString: string;
}) => {
  return new Pool({
    connectionString,
  });
};

export type PostgresPool = Pool;
