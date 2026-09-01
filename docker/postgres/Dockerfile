# 在现有 Postgres 15 镜像上编译 pgvector，避免再开一个占用 5432 的容器。
FROM postgres:15-alpine

RUN apk add --no-cache --virtual .build-deps git build-base clang19 llvm19-dev \
    && git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector \
    && make -C /tmp/pgvector install \
    && rm -rf /tmp/pgvector \
    && apk del .build-deps
