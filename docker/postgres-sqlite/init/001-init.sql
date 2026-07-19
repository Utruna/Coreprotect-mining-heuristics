CREATE EXTENSION IF NOT EXISTS sqlite_fdw;

CREATE SCHEMA IF NOT EXISTS coreprotect;

CREATE SERVER IF NOT EXISTS coreprotect_sqlite
  FOREIGN DATA WRAPPER sqlite_fdw
  OPTIONS (database '/sqlite/database.db');