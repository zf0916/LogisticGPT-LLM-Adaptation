#!/bin/bash
set -e

# Create dvdrental database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE dvdrental;
EOSQL

# Restore database from mounted file
pg_restore -U "$POSTGRES_USER" -d dvdrental /tmp/dvdrental.tar