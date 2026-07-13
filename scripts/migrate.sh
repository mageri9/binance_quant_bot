#!/usr/bin/env bash
# Usage:
#   ./scripts/migrate.sh upgrade head       — apply all migrations
#   ./scripts/migrate.sh revision -m "msg"  — create new migration
#   ./scripts/migrate.sh downgrade -1       — roll back one step

set -e
docker compose run --rm bot alembic "$@"
