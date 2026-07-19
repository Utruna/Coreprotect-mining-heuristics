FROM postgres:17-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libsqlite3-dev \
        postgresql-server-dev-17 \
        sqlite3 \
    && git clone --depth 1 https://github.com/pgspider/sqlite_fdw.git /tmp/sqlite_fdw \
    && cd /tmp/sqlite_fdw \
    && make USE_PGXS=1 \
    && make USE_PGXS=1 install \
    && rm -rf /var/lib/apt/lists/*

COPY docker/postgres-sqlite/init/001-init.sql /docker-entrypoint-initdb.d/001-init.sql