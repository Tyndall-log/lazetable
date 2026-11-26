"""장벽 단계 결과 저장소(빌드 중간 결과).

중간 노드에서 묶음을 다시 만들려면 앞 단계 결과가 모두 있어야 한다. 그 결과를 메모리에 쌓지 않고
단위(앞 단계의 묶음)마다 끝나는 대로 디스크에 쓴다. 메타 값(JSON)은 묶음 계획에 쓰고, 행 전체 값은
pickle로 두었다가 묶음을 실행할 때 그 구성원만 읽는다. 단위별로 저장하므로 빌드가 멈춰도 끝난
단위는 다시 계산하지 않는다(재개 지점).

SQLite는 디스크가 기본이고 최근 페이지만 캐시한다. 캐시 상한은 ``cache_mb``로 정한다.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .grouping import Group, ItemMeta, PlannedGroup


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StageStore:
    def __init__(self, path: Path, *, cache_mb: int = 256) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(f"PRAGMA cache_size={-max(1, int(cache_mb)) * 1024}")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stage_units (
                stage INTEGER NOT NULL,
                unit_idx INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (stage, unit_idx)
            );
            CREATE TABLE IF NOT EXISTS stage_items (
                stage INTEGER NOT NULL,
                unit_idx INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                sources_json TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                value BLOB NOT NULL,
                PRIMARY KEY (stage, unit_idx, ordinal)
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def item_id(stage: int, unit_idx: int, ordinal: int) -> str:
        return f"{stage}:{unit_idx}:{ordinal}"

    def completed_units(self, stage: int) -> set[int]:
        return {
            int(row[0])
            for row in self._connection.execute(
                "SELECT unit_idx FROM stage_units WHERE stage=? AND status='completed'", (stage,)
            )
        }

    def clear_failed(self, stage: int) -> None:
        self._connection.execute("DELETE FROM stage_units WHERE stage=? AND status='failed'", (stage,))
        self._connection.commit()

    def commit_unit(
        self,
        stage: int,
        unit_idx: int,
        items: Sequence[tuple[dict[str, Any], Sequence[int], Any]],
    ) -> None:
        """단위 하나의 결과(메타 값, 원본 출처, 행 값)를 한 트랜잭션으로 쓴다."""

        with self._connection:
            self._connection.execute(
                "DELETE FROM stage_items WHERE stage=? AND unit_idx=?", (stage, unit_idx)
            )
            self._connection.executemany(
                "INSERT INTO stage_items VALUES(?, ?, ?, ?, ?, ?)",
                [
                    (
                        stage,
                        unit_idx,
                        ordinal,
                        json.dumps(list(sources)),
                        json.dumps(meta, ensure_ascii=False, sort_keys=True),
                        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    for ordinal, (meta, sources, value) in enumerate(items)
                ],
            )
            self._connection.execute(
                "INSERT INTO stage_units VALUES(?, ?, 'completed', NULL, ?) "
                "ON CONFLICT(stage, unit_idx) DO UPDATE SET status='completed', error=NULL, "
                "updated_at=excluded.updated_at",
                (stage, unit_idx, _now()),
            )

    def mark_failed(self, stage: int, unit_idx: int, error: BaseException) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO stage_units VALUES(?, ?, 'failed', ?, ?) "
                "ON CONFLICT(stage, unit_idx) DO UPDATE SET status='failed', error=excluded.error, "
                "updated_at=excluded.updated_at",
                (stage, unit_idx, f"{type(error).__name__}: {error}", _now()),
            )

    def item_metas(self, stage: int) -> list[ItemMeta]:
        """단계 결과의 메타 값(단위·순번 순서). 무거운 값은 읽지 않는다."""

        return [
            ItemMeta(
                id=self.item_id(stage, int(row[0]), int(row[1])),
                values=json.loads(row[3]),
                sources=tuple(json.loads(row[2])),
            )
            for row in self._connection.execute(
                "SELECT unit_idx, ordinal, sources_json, meta_json FROM stage_items "
                "WHERE stage=? ORDER BY unit_idx, ordinal",
                (stage,),
            )
        ]

    def load(self, item_id: str) -> Any:
        stage, unit_idx, ordinal = (int(part) for part in item_id.split(":"))
        row = self._connection.execute(
            "SELECT value FROM stage_items WHERE stage=? AND unit_idx=? AND ordinal=?",
            (stage, unit_idx, ordinal),
        ).fetchone()
        if row is None:
            raise KeyError(f"stage item not found: {item_id}")
        return pickle.loads(row[0])


class StageUnits(Sequence[Group]):
    """장벽 뒤 단계의 단위(묶음) 목록. 묶음 구성원 값은 꺼낼 때 디스크에서 읽는다."""

    def __init__(self, store: StageStore, groups: Sequence[PlannedGroup]) -> None:
        self._store = store
        self._groups = list(groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        planned = self._groups[index]
        return Group(
            [self._store.load(member_id) for member_id in planned.member_ids],
            key=planned.key,
            member_ids=planned.member_ids,
            sources=planned.sources,
        )

    def sources(self, index: int) -> tuple[int, ...]:
        return self._groups[index].sources
