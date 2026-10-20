#!/bin/bash
# Runs once, when the Postgres data volume is first initialized. On an existing volume,
# run the same statements by hand (see README).
set -euo pipefail

: "${MLFLOW_DB_PASSWORD:?MLFLOW_DB_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 -v pw="$MLFLOW_DB_PASSWORD" \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'EOSQL'
CREATE ROLE mlflow LOGIN PASSWORD :'pw';
CREATE DATABASE mlflow OWNER mlflow;
EOSQL
