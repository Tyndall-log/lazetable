"""SQLite 기반 LazeTable artifact metadata 저장소."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence


SQLITE_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode(value: str | None, default: Any) -> Any:
    return default if value is None else json.loads(value)


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    field_name: str
    payload_id: str
    relative_path: str
    status: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    artifact_name: str
    artifact_version: str
    build_signature: str
    phase: str
    status: str
    source_row_count: int
    output_row_count: int
    content_hash: str | None
    metadata: dict[str, Any]


class SQLiteArtifactStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self, *, read_only: bool = False, immutable: bool = False) -> sqlite3.Connection:
        if read_only:
            suffix = "?mode=ro"
            if immutable:
                suffix += "&immutable=1"
            connection = sqlite3.connect(
                f"file:{self.path.as_posix()}{suffix}",
                uri=True,
                timeout=30.0,
            )
            connection.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize_schema(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifact (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    artifact_name TEXT NOT NULL,
                    artifact_version TEXT NOT NULL,
                    build_signature TEXT NOT NULL,
                    signature_payload_json TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_row_count INTEGER NOT NULL DEFAULT 0,
                    output_row_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS dependencies (
                    position INTEGER PRIMARY KEY,
                    artifact_name TEXT NOT NULL,
                    artifact_version TEXT NOT NULL,
                    manifest_hash TEXT,
                    join_json TEXT
                );

                CREATE TABLE IF NOT EXISTS source_rows (
                    source_idx INTEGER PRIMARY KEY,
                    source_key TEXT,
                    source_hash TEXT,
                    status TEXT NOT NULL,
                    output_count INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS raw_output_rows (
                    source_idx INTEGER NOT NULL,
                    output_idx INTEGER NOT NULL,
                    unique_key TEXT NOT NULL UNIQUE,
                    row_json TEXT NOT NULL,
                    PRIMARY KEY (source_idx, output_idx),
                    FOREIGN KEY (source_idx) REFERENCES source_rows(source_idx)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS payloads (
                    source_idx INTEGER NOT NULL,
                    output_idx INTEGER NOT NULL,
                    field_name TEXT NOT NULL,
                    payload_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY (source_idx, output_idx, field_name),
                    FOREIGN KEY (source_idx, output_idx)
                        REFERENCES raw_output_rows(source_idx, output_idx)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS failures (
                    source_idx INTEGER PRIMARY KEY,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    FOREIGN KEY (source_idx) REFERENCES source_rows(source_idx)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS final_output_rows (
                    row_order INTEGER PRIMARY KEY,
                    unique_key TEXT NOT NULL UNIQUE,
                    row_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS statistics (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                );

                CREATE INDEX IF NOT EXISTS source_rows_status_idx
                    ON source_rows(status);
                CREATE INDEX IF NOT EXISTS payloads_relative_path_idx
                    ON payloads(relative_path);
                """
            )
            connection.execute(
                "INSERT INTO schema_info(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SQLITE_SCHEMA_VERSION),),
            )

    def validate_schema(self, *, read_only: bool = False) -> None:
        connection = self._connect(read_only=read_only, immutable=read_only)
        try:
            row = connection.execute(
                "SELECT value FROM schema_info WHERE key='schema_version'"
            ).fetchone()
            if row is None or int(row["value"]) != SQLITE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported SQLite artifact schema: {None if row is None else row['value']}"
                )
        finally:
            connection.close()

    def initialize_build(
        self,
        *,
        artifact_name: str,
        artifact_version: str,
        build_signature: str,
        signature_payload: dict[str, Any],
        dependencies: Sequence[dict[str, Any]],
    ) -> None:
        self.initialize_schema()
        now = _now()
        with self.transaction() as connection:
            existing = connection.execute("SELECT build_signature FROM artifact").fetchone()
            if existing is not None and existing["build_signature"] != build_signature:
                raise ValueError("partial artifact signature mismatch")
            connection.execute(
                """
                INSERT INTO artifact(
                    singleton, artifact_name, artifact_version, build_signature,
                    signature_payload_json, phase, status, created_at, updated_at
                ) VALUES(1, ?, ?, ?, ?, 'preparing', 'building', ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (
                    artifact_name,
                    artifact_version,
                    build_signature,
                    _encode(signature_payload),
                    now,
                    now,
                ),
            )
            connection.execute("DELETE FROM dependencies")
            connection.executemany(
                """
                INSERT INTO dependencies(
                    position, artifact_name, artifact_version, manifest_hash, join_json
                ) VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        position,
                        item["artifact_name"],
                        item["artifact_version"],
                        item.get("manifest_hash"),
                        _encode(item.get("join")) if item.get("join") is not None else None,
                    )
                    for position, item in enumerate(dependencies)
                ],
            )

    def artifact(self, *, read_only: bool = False) -> StoredArtifact | None:
        connection = self._connect(read_only=read_only, immutable=read_only)
        try:
            row = connection.execute("SELECT * FROM artifact WHERE singleton=1").fetchone()
            if row is None:
                return None
            return StoredArtifact(
                artifact_name=row["artifact_name"],
                artifact_version=row["artifact_version"],
                build_signature=row["build_signature"],
                phase=row["phase"],
                status=row["status"],
                source_row_count=int(row["source_row_count"]),
                output_row_count=int(row["output_row_count"]),
                content_hash=row["content_hash"],
                metadata=_decode(row["metadata_json"], {}),
            )
        finally:
            connection.close()

    def set_phase(self, phase: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE artifact SET phase=?, updated_at=? WHERE singleton=1",
                (phase, _now()),
            )

    def ensure_source_rows(self, count: int) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO source_rows(source_idx, status) VALUES(?, 'pending')",
                [(index,) for index in range(count)],
            )
            connection.execute(
                "UPDATE artifact SET source_row_count=?, phase='transforming', updated_at=? "
                "WHERE singleton=1",
                (count, _now()),
            )

    def completed_sources(self) -> set[int]:
        connection = self._connect(read_only=True)
        try:
            return {
                int(row["source_idx"])
                for row in connection.execute(
                    "SELECT source_idx FROM source_rows WHERE status='completed'"
                )
            }
        finally:
            connection.close()

    def failed_sources(self) -> set[int]:
        connection = self._connect(read_only=True)
        try:
            return {
                int(row["source_idx"])
                for row in connection.execute(
                    "SELECT source_idx FROM source_rows WHERE status='failed'"
                )
            }
        finally:
            connection.close()

    def clear_failure(self, source_idx: int) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM failures WHERE source_idx=?", (source_idx,))
            connection.execute(
                "UPDATE source_rows SET status='pending' WHERE source_idx=?", (source_idx,)
            )

    def commit_source(
        self,
        source_idx: int,
        rows: Sequence[tuple[int, str, dict[str, Any], Sequence[PayloadRecord]]],
    ) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute("DELETE FROM raw_output_rows WHERE source_idx=?", (source_idx,))
            for output_idx, unique_key, row, payload_records in rows:
                connection.execute(
                    "INSERT INTO raw_output_rows VALUES(?, ?, ?, ?)",
                    (source_idx, output_idx, unique_key, _encode(row)),
                )
                connection.executemany(
                    "INSERT INTO payloads VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            source_idx,
                            output_idx,
                            payload.field_name,
                            payload.payload_id,
                            payload.relative_path,
                            payload.status,
                            payload.size_bytes,
                            payload.sha256,
                        )
                        for payload in payload_records
                    ],
                )
            connection.execute("DELETE FROM failures WHERE source_idx=?", (source_idx,))
            connection.execute(
                """
                UPDATE source_rows
                SET status='completed', output_count=?, attempts=attempts+1,
                    completed_at=?, started_at=COALESCE(started_at, ?)
                WHERE source_idx=?
                """,
                (len(rows), now, now, source_idx),
            )

    def mark_failed(self, source_idx: int, error: BaseException) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE source_rows SET status='failed', attempts=attempts+1, "
                "started_at=COALESCE(started_at, ?) WHERE source_idx=?",
                (now, source_idx),
            )
            connection.execute(
                """
                INSERT INTO failures VALUES(?, ?, ?, ?)
                ON CONFLICT(source_idx) DO UPDATE SET
                    error_type=excluded.error_type,
                    message=excluded.message,
                    occurred_at=excluded.occurred_at
                """,
                (source_idx, type(error).__name__, str(error), now),
            )

    def raw_rows(self) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True)
        try:
            return [
                _decode(row["row_json"], {})
                for row in connection.execute(
                    "SELECT row_json FROM raw_output_rows ORDER BY source_idx, output_idx"
                )
            ]
        finally:
            connection.close()

    def finalize(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        metadata: dict[str, Any],
        statistics: dict[str, dict[str, Any]],
        content_hash: str,
    ) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute("DELETE FROM final_output_rows")
            connection.execute("DELETE FROM statistics")
            connection.executemany(
                "INSERT INTO final_output_rows VALUES(?, ?, ?)",
                [
                    (index, _encode(row.get("__unique_key__", index)), _encode(row))
                    for index, row in enumerate(rows)
                ],
            )
            connection.executemany(
                "INSERT INTO statistics VALUES(?, ?, ?)",
                [
                    (namespace, key, _encode(value))
                    for namespace, values in statistics.items()
                    for key, value in values.items()
                ],
            )
            connection.execute(
                """
                UPDATE artifact
                SET phase='completed', status='completed', output_row_count=?,
                    content_hash=?, metadata_json=?, updated_at=?, completed_at=?
                WHERE singleton=1
                """,
                (len(rows), content_hash, _encode(metadata), now, now),
            )

    def final_rows(self) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True, immutable=True)
        try:
            return [
                _decode(row["row_json"], {})
                for row in connection.execute(
                    "SELECT row_json FROM final_output_rows ORDER BY row_order"
                )
            ]
        finally:
            connection.close()

    def statistics(self) -> dict[str, dict[str, Any]]:
        connection = self._connect(read_only=True, immutable=True)
        try:
            result: dict[str, dict[str, Any]] = {}
            for row in connection.execute(
                "SELECT namespace, key, value_json FROM statistics ORDER BY namespace, key"
            ):
                result.setdefault(row["namespace"], {})[row["key"]] = _decode(
                    row["value_json"], None
                )
            return result
        finally:
            connection.close()

    def checkpoint_and_close(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
