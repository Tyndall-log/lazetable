"""노드 입력 묶음(n:1·n:n).

노드는 ``Node(..., groups=None)``이면 앞 결과를 한 행씩, ``groups=...``이면 앞 결과를 묶어서 받는다.
노드 함수가 받는 원소는 항상 ``Group``(목록처럼 ``len``·순회·인덱스, ``.key``)이다.

- 첫 노드의 묶음은 실행 전(계획 단계)에 원본 행으로 정한다.
- 중간 노드의 묶음은 앞 단계 결과가 모두 디스크에 저장된 뒤 그 메타 값으로 정한다(장벽).
- 묶음 함수는 메타 값(스칼라와 그 목록·사전)만 본다. 메쉬 같은 무거운 값은 묶음을 실행할 때 읽는다.
- 같은 행이 여러 묶음에 들어갈 수 있다. 키를 생략하면 구성원 ID 목록의 해시가 키다. 같은 키가 두 번
  나오거나, 빈 묶음, 한 묶음에 같은 행이 두 번 들어가면 오류다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class GroupingError(ValueError):
    """묶음 계획이 규칙을 어길 때 발생한다."""


@dataclass(frozen=True)
class ItemMeta:
    """묶음 함수가 보는 행 하나. ``values``는 메타 값만(무거운 값 제외)."""

    id: str
    values: Mapping[str, Any]
    sources: tuple[int, ...]

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


class Group(Sequence[Any]):
    """노드 입력 묶음. 목록처럼 쓰고 ``.key``로 키를 읽는다.

    묶음 함수에서 만들 때는 ``Group(구성원 ItemMeta 목록, key=...)``로 쓴다. ``key``를 생략하면 구성원
    ID 해시가 키가 된다. 실행할 때 노드가 받는 묶음의 원소는 실제 행 값이다.
    """

    __slots__ = ("_members", "key", "member_ids", "sources")

    def __init__(
        self,
        members: Iterable[Any],
        key: Any = None,
        *,
        member_ids: Sequence[str] = (),
        sources: Sequence[int] = (),
    ) -> None:
        self._members = list(members)
        self.key = key
        self.member_ids = tuple(member_ids)
        self.sources = tuple(sources)

    @classmethod
    def single(cls, value: Any, *, member_id: str = "", sources: Sequence[int] = ()) -> "Group":
        return cls([value], key=member_id or None, member_ids=(member_id,) if member_id else (),
                   sources=sources)

    def __len__(self) -> int:
        return len(self._members)

    def __getitem__(self, index):  # type: ignore[override]
        return self._members[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._members)

    def __repr__(self) -> str:
        return f"Group(key={self.key!r}, size={len(self._members)})"


def _jsonable(value: Any) -> Any:
    """메타 값으로 쓸 수 있으면 JSON 값, 아니면 ``_SKIP``."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return _jsonable(item())
    if isinstance(value, (list, tuple)):
        out = [_jsonable(v) for v in value]
        return _SKIP if any(v is _SKIP for v in out) else out
    if isinstance(value, Mapping):
        out_map = {str(k): _jsonable(v) for k, v in value.items()}
        return _SKIP if any(v is _SKIP for v in out_map.values()) else out_map
    return _SKIP


_SKIP = object()


def meta_values(row: Any) -> dict[str, Any]:
    """행에서 메타 값만 뽑는다(JSON으로 나타낼 수 있는 값). ``ParsedRow``는 원문 값을 쓴다."""

    values = row.as_dict() if hasattr(row, "as_dict") else row
    if not isinstance(values, Mapping):
        raise GroupingError(f"묶을 행은 dict여야 함: {type(row).__name__}")
    out: dict[str, Any] = {}
    for name, value in values.items():
        converted = _jsonable(value)
        if converted is not _SKIP:
            out[str(name)] = converted
    return out


def key_text(key: Any) -> str:
    return json.dumps(_jsonable(key) if _jsonable(key) is not _SKIP else repr(key),
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GroupSpec:
    """묶음 규칙. ``columns``는 키 계산에 쓰는 열(모르면 None). 서명에는 ``name``과 ``columns``가 들어간다."""

    name: str
    fn: Callable[[Sequence[ItemMeta]], Iterable[Group]] = field(compare=False)
    columns: tuple[str, ...] | None = None

    def signature(self) -> dict[str, Any]:
        return {"name": self.name, "columns": None if self.columns is None else list(self.columns)}


def as_group_spec(groups: Any) -> GroupSpec | None:
    if groups is None or isinstance(groups, GroupSpec):
        return groups
    if callable(groups):
        name = f"{getattr(groups, '__module__', '')}.{getattr(groups, '__qualname__', repr(groups))}"
        return GroupSpec(name=name, fn=groups, columns=None)
    raise TypeError(f"groups must be None, a GroupSpec or a callable: {type(groups).__name__}")


class Grouping:
    """빌더의 ``self.grouping``. 자주 쓰는 묶음 규칙을 만든다."""

    def by_columns(self, *columns: str) -> GroupSpec:
        """열 값이 같은 행끼리 묶는다. 키는 열 하나면 그 값, 여럿이면 튜플."""

        if not columns:
            raise ValueError("by_columns needs at least one column")
        names = tuple(str(c) for c in columns)

        def fn(items: Sequence[ItemMeta]) -> list[Group]:
            buckets: dict[str, tuple[Any, list[ItemMeta]]] = {}
            for item in items:
                missing = [c for c in names if c not in item.values]
                if missing:
                    raise GroupingError(f"{item.id}: 묶음 열이 없거나 메타 값이 아님: {missing}")
                key = item.values[names[0]] if len(names) == 1 else tuple(item.values[c] for c in names)
                buckets.setdefault(key_text(key), (key, []))[1].append(item)
            return [Group(members, key=key) for key, members in buckets.values()]

        return GroupSpec(name=f"by_columns{list(names)}", fn=fn, columns=names)

    def by_list_column(self, column: str) -> GroupSpec:
        """목록 열의 값마다 그 묶음에 넣는다(한 행이 여러 묶음에 들어감)."""

        name = str(column)

        def fn(items: Sequence[ItemMeta]) -> list[Group]:
            buckets: dict[str, tuple[Any, list[ItemMeta]]] = {}
            for item in items:
                values = item.values.get(name)
                if not isinstance(values, list):
                    raise GroupingError(f"{item.id}: {name!r}는 목록 메타 값이어야 함")
                for key in dict.fromkeys(key_text(v) for v in values):
                    value = json.loads(key)
                    buckets.setdefault(key, (value, []))[1].append(item)
            return [Group(members, key=key) for key, members in buckets.values()]

        return GroupSpec(name=f"by_list_column[{name!r}]", fn=fn, columns=(name,))

    def by_key(self, key: Callable[[ItemMeta], Any], *, uses: Sequence[str] | None = None) -> GroupSpec:
        """``key(item)``이 같은 행끼리 묶는다. ``uses``(키 계산에 쓰는 열)를 빼면 장벽으로 처리한다."""

        def fn(items: Sequence[ItemMeta]) -> list[Group]:
            buckets: dict[str, tuple[Any, list[ItemMeta]]] = {}
            for item in items:
                value = key(item)
                buckets.setdefault(key_text(value), (value, []))[1].append(item)
            return [Group(members, key=value) for value, members in buckets.values()]

        name = f"by_key[{getattr(key, '__module__', '')}.{getattr(key, '__qualname__', repr(key))}]"
        return GroupSpec(name=name, fn=fn, columns=None if uses is None else tuple(str(u) for u in uses))


@dataclass(frozen=True)
class PlannedGroup:
    key: Any
    member_ids: tuple[str, ...]
    sources: tuple[int, ...]


def plan_groups(spec: GroupSpec, items: Sequence[ItemMeta]) -> tuple[list[PlannedGroup], str]:
    """묶음 규칙을 적용해 (묶음 목록, 계획 해시)를 낸다. 규칙 위반은 ``GroupingError``."""

    by_id = {item.id: item for item in items}
    planned: list[PlannedGroup] = []
    seen: dict[str, int] = {}
    for position, group in enumerate(spec.fn(items)):
        if not isinstance(group, Group):
            raise GroupingError(f"{spec.name}: 묶음 함수는 Group을 내야 함: {type(group).__name__}")
        ids: list[str] = []
        for member in group:
            member_id = member.id if isinstance(member, ItemMeta) else str(member)
            if member_id not in by_id:
                raise GroupingError(f"{spec.name}: 입력에 없는 구성원 {member_id!r}")
            ids.append(member_id)
        if not ids:
            raise GroupingError(f"{spec.name}: 빈 묶음(위치 {position})")
        if len(set(ids)) != len(ids):
            raise GroupingError(f"{spec.name}: 한 묶음에 같은 구성원이 두 번 들어감(위치 {position})")
        key = group.key
        if key is None:
            key = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:24]
        text = key_text(key)
        if text in seen:
            raise GroupingError(f"{spec.name}: 같은 키가 두 번 나옴: {key!r}")
        seen[text] = position
        sources = sorted({s for member_id in ids for s in by_id[member_id].sources})
        planned.append(PlannedGroup(key=key, member_ids=tuple(ids), sources=tuple(sources)))
    digest = hashlib.sha256()
    for group in planned:
        digest.update(key_text(group.key).encode())
        digest.update(b"\0")
        digest.update("\n".join(group.member_ids).encode())
        digest.update(b"\1")
    return planned, digest.hexdigest()
