#!/usr/bin/env python3
"""Wipe every LightRAG-managed datum across Postgres, Neo4j, Qdrant, and Redis.

Intended for the per-request-workspace migration cutover — see
docs/MIGRATION_TO_PER_REQUEST_WORKSPACE.md (or the PR description for the
workspace work) for the surrounding deploy procedure.

WHAT THIS SCRIPT TOUCHES
    - Postgres: drops and recreates the `lightrag` database
    - Neo4j: DETACH DELETE every node
    - Qdrant: deletes every collection
    - Redis: deletes every LightRAG-owned key

WHAT THIS SCRIPT PRESERVES IN REDIS
    Backend-owned namespaces stay intact:
      - `ingestion:*`     application ingestion tracking
      - `schema:*`        application schema cache
      - `arq:*`           async task queue state
    The backend does not use the `llm_response_cache:*` namespace (grep
    confirms), and post-workspace-migration LightRAG only ever writes to
    `<workspace>_llm_response_cache:*` — so any unprefixed
    `llm_response_cache:*` keys are pre-migration orphans and are deleted
    along with the workspace-prefixed copies.

USAGE
    # Dry run (default — prints what would change, no writes)
    python scripts/wipe_storage.py

    # Actually wipe
    python scripts/wipe_storage.py --execute

    # Wipe only one backend (handy when retrying after a partial failure)
    python scripts/wipe_storage.py --only postgres --execute

CREDENTIALS
    Read from .env in the current working directory by default. Override
    individual values via environment variables before running.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

# --- env loader (lightweight; avoids requiring python-dotenv) -----------------


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# --- backend wipers -----------------------------------------------------------


def wipe_postgres(execute: bool) -> str:
    """Drop and recreate the `lightrag` database. LightRAG will recreate the
    schema on its next startup via the storage-init migration path.
    """
    import psycopg2  # type: ignore
    from psycopg2 import sql

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    dbname = os.environ.get("POSTGRES_DATABASE", "lightrag")

    if not execute:
        return f"would DROP/CREATE database {dbname!r} on {host}:{port}"

    # We need to connect to a DB other than the one we're dropping.
    conn = psycopg2.connect(
        host=host, port=port, user=user, password=password, dbname="postgres"
    )
    conn.autocommit = True  # CREATE/DROP DATABASE can't run inside a transaction
    try:
        cur = conn.cursor()
        # Terminate stragglers so DROP DATABASE doesn't error with
        # "database is being accessed by other users".
        cur.execute(
            sql.SQL(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()"
            ),
            (dbname,),
        )
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname)))
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    finally:
        conn.close()
    return f"dropped and recreated {dbname!r}"


def wipe_neo4j(execute: bool) -> str:
    """Detach-delete every node in the configured Neo4j database."""
    from neo4j import GraphDatabase  # type: ignore

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    if not execute:
        return f"would DETACH DELETE all nodes in Neo4j db {database!r} at {uri}"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            before = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            session.run("MATCH (n) DETACH DELETE n")
            after = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        return f"Neo4j: deleted {before - after} nodes (was {before}, now {after})"
    finally:
        driver.close()


def wipe_qdrant(execute: bool) -> str:
    """Drop every Qdrant collection. LightRAG will recreate the ones it needs
    on next startup using its workspace-aware naming.
    """
    from qdrant_client import QdrantClient  # type: ignore

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None

    client = QdrantClient(url=url, api_key=api_key)
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as e:
        return f"Qdrant: failed to list collections ({e})"

    if not execute:
        return f"would delete {len(existing)} Qdrant collection(s): {existing}"

    deleted = []
    for name in existing:
        try:
            client.delete_collection(name)
            deleted.append(name)
        except Exception as e:
            print(f"  [WARN] Qdrant: failed to drop {name}: {e}", file=sys.stderr)
    return f"Qdrant: dropped {len(deleted)}/{len(existing)} collection(s)"


_REDIS_LIGHTRAG_KEY_PATTERNS = (
    # Legacy unprefixed namespaces that exist when LightRAG ran with an
    # empty workspace before per-workspace prefixing was introduced.
    "entity_chunks:*",
    "relation_chunks:*",
    "full_entities:*",
    "full_relations:*",
    "text_chunks:*",
    "full_docs:doc-*",
    # The unprefixed LLM cache only exists as a pre-workspace-migration
    # orphan — current LightRAG always writes under <workspace>_… and the
    # backend doesn't use this namespace. Pattern uses `:` as separator
    # so workspace-prefixed keys like `<ws>_llm_response_cache:*` (which
    # don't start with `llm_response_cache:`) are not matched here.
    "llm_response_cache:*",
    # Anything LightRAG-prefixed by a workspace name follows the shape
    # `<workspace>_<namespace>:...`. Iterating known workspaces from env keeps
    # this surgical; passing --include-all-workspaces sweeps every key with
    # the LightRAG namespace suffix regardless of prefix.
)


def _scan_keys(redis_client, pattern: str) -> Iterable[str]:
    cursor = 0
    while True:
        cursor, batch = redis_client.scan(cursor=cursor, match=pattern, count=1000)
        for key in batch:
            yield key.decode() if isinstance(key, bytes) else key
        if cursor == 0:
            return


def wipe_redis(execute: bool, include_all_workspaces: bool) -> str:
    """Delete LightRAG-owned keys, preserving backend state.

    Two sweep modes:
        - surgical (default): only fixed legacy patterns + the workspace
          patterns enumerated below
        - --include-all-workspaces: also matches any key whose suffix is one
          of LightRAG's known namespaces (handles unknown workspaces)
    """
    import redis  # type: ignore

    url = os.environ.get("REDIS_URI") or os.environ.get("REDIS_URL")
    if not url:
        return "Redis: no REDIS_URI/REDIS_URL set, skipping"

    client = redis.from_url(url)
    patterns = list(_REDIS_LIGHTRAG_KEY_PATTERNS)

    # Add patterns for any known workspaces. Read WORKSPACE from env plus
    # any extras passed via --workspaces.
    known = set()
    env_ws = os.environ.get("WORKSPACE", "").strip()
    if env_ws:
        known.add(env_ws)
    for ws in _extra_workspaces:
        if ws:
            known.add(ws)
    for ws in sorted(known):
        # LightRAG storage namespaces (see storage_namespace.py upstream)
        for ns in (
            "doc_status",
            "full_docs",
            "text_chunks",
            "entity_chunks",
            "relation_chunks",
            "full_entities",
            "full_relations",
            "llm_response_cache",
        ):
            patterns.append(f"{ws}_{ns}:*")

    if include_all_workspaces:
        # Broad sweep: any key ending in a LightRAG namespace, regardless
        # of the (unknown) workspace prefix. Suffixes are unique to
        # LightRAG so collisions with backend keys are unlikely.
        for ns in (
            "doc_status",
            "full_docs",
            "text_chunks",
            "entity_chunks",
            "relation_chunks",
            "full_entities",
            "full_relations",
        ):
            patterns.append(f"*_{ns}:*")

    if not execute:
        sample = {}
        for pat in patterns:
            count = sum(1 for _ in _scan_keys(client, pat))
            sample[pat] = count
        total = sum(sample.values())
        return f"would delete {total} Redis key(s) across {len(patterns)} pattern(s); {sample}"

    deleted = 0
    for pat in patterns:
        batch = []
        for key in _scan_keys(client, pat):
            batch.append(key)
            if len(batch) >= 500:
                deleted += client.delete(*batch)
                batch = []
        if batch:
            deleted += client.delete(*batch)
    return f"Redis: deleted {deleted} key(s)"


# --- CLI ---------------------------------------------------------------------


_extra_workspaces: list[str] = []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually wipe. Default is dry-run, which only reports.",
    )
    parser.add_argument(
        "--only",
        choices=["postgres", "neo4j", "qdrant", "redis"],
        action="append",
        help="Limit to one backend. Repeatable. Default: all four.",
    )
    parser.add_argument(
        "--workspaces",
        nargs="*",
        default=[],
        help="Extra workspace names whose Redis keys should be deleted. "
        "The value of $WORKSPACE is included automatically.",
    )
    parser.add_argument(
        "--include-all-workspaces",
        action="store_true",
        help="Also delete Redis keys for any unknown workspace whose suffix matches "
        "a LightRAG namespace. Use after multi-workspace deployments.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to env file (default: ./.env)",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    global _extra_workspaces
    _extra_workspaces = list(args.workspaces)

    backends = args.only or ["postgres", "neo4j", "qdrant", "redis"]

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"=== {mode} | backends: {', '.join(backends)} ===\n")

    if not args.execute:
        print("(no writes will happen — pass --execute to actually wipe)\n")

    results = {}
    started = time.time()

    if "postgres" in backends:
        print("[postgres] ", end="", flush=True)
        try:
            results["postgres"] = wipe_postgres(args.execute)
        except Exception as e:
            results["postgres"] = f"ERROR: {e}"
        print(results["postgres"])

    if "neo4j" in backends:
        print("[neo4j]    ", end="", flush=True)
        try:
            results["neo4j"] = wipe_neo4j(args.execute)
        except Exception as e:
            results["neo4j"] = f"ERROR: {e}"
        print(results["neo4j"])

    if "qdrant" in backends:
        print("[qdrant]   ", end="", flush=True)
        try:
            results["qdrant"] = wipe_qdrant(args.execute)
        except Exception as e:
            results["qdrant"] = f"ERROR: {e}"
        print(results["qdrant"])

    if "redis" in backends:
        print("[redis]    ", end="", flush=True)
        try:
            results["redis"] = wipe_redis(args.execute, args.include_all_workspaces)
        except Exception as e:
            results["redis"] = f"ERROR: {e}"
        print(results["redis"])

    elapsed = time.time() - started
    print(f"\ndone in {elapsed:.1f}s")
    return 0 if all(not str(v).startswith("ERROR") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
