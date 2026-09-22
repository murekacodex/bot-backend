from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import fcntl


def _database_url() -> str | None:
    from app.config import get_settings

    return get_settings().database_url


def _state_key(path: Path) -> str:
    return path.name


def _database_connection():
    import psycopg

    return psycopg.connect(_database_url())


def _ensure_state_table(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    connection.commit()


def read_json(path: Path, default: Any) -> Any:
    """Read shared state, seeding Postgres once from the legacy JSON file."""
    if _database_url():
        with _database_connection() as connection:
            _ensure_state_table(connection)
            row = connection.execute("SELECT payload FROM app_state WHERE key = %s", (_state_key(path),)).fetchone()
            if row is not None:
                return row[0]
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except (OSError, json.JSONDecodeError):
            return default
        if path.exists():
            atomic_write_json(path, payload)
        return payload
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    if _database_url():
        with _database_connection() as connection:
            _ensure_state_table(connection)
            connection.execute("SELECT pg_advisory_lock(hashtext(%s))", (_state_key(path),))
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (_state_key(path),))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: Path, payload: Any) -> None:
    if _database_url():
        import psycopg

        with _database_connection() as connection:
            _ensure_state_table(connection)
            connection.execute(
                """
                INSERT INTO app_state (key, payload, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (_state_key(path), psycopg.types.json.Jsonb(payload)),
            )
            connection.commit()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
