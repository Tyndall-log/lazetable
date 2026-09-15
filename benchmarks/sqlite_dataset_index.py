from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from lazetable import LazyTable
from lazetable.sqlite_store import SQLiteArtifactStore


BENCHMARK_ROOT = Path(__file__).resolve().parent / ".data" / "sqlite_dataset_index"


def measure(action, *, repeat: int = 5):
    timings = []
    result = None
    for _ in range(repeat):
        started = time.perf_counter()
        result = action()
        timings.append(time.perf_counter() - started)
    return min(timings), result


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite DatasetIndex 순회 성능을 측정합니다")
    parser.add_argument("--rows", type=int, default=25_000)
    args = parser.parse_args()
    if BENCHMARK_ROOT.exists():
        shutil.rmtree(BENCHMARK_ROOT)
    BENCHMARK_ROOT.mkdir(parents=True)
    rows = [
        {
            "__unique_key__": [f"row-{index}"],
            "id": f"row-{index}",
            "split": "validation" if index % 10 == 0 else "train",
            "fdi": index % 32,
            "payload_ref": f"payload/row-{index}/data.npz",
        }
        for index in range(args.rows)
    ]
    store = SQLiteArtifactStore(BENCHMARK_ROOT / "artifact.sqlite")
    store.initialize_build(
        artifact_name="benchmark",
        artifact_version="v1",
        build_signature="benchmark",
        signature_payload={},
        dependencies=[],
    )
    store.finalize(
        rows=rows,
        metadata={"artifact_roots": {"default": str(BENCHMARK_ROOT.parent)}},
        statistics={},
        content_hash="benchmark",
    )
    store.checkpoint_and_close()

    open_seconds, table = measure(lambda: LazyTable.open(BENCHMARK_ROOT))
    index_seconds, index = measure(
        lambda: table.to_dataset_index(
            where={"split": "train"},
            fields=["id", "fdi", "payload_ref"],
        )
    )
    order = list(range(len(index)))
    random.Random(42).shuffle(order)
    iteration_seconds, _ = measure(lambda: [index[item].raw("fdi") for item in order])
    print(
        json.dumps(
            {
                "row_count": args.rows,
                "train_row_count": len(index),
                "sqlite_size_mib": store.path.stat().st_size / (1024 * 1024),
                "open_seconds": open_seconds,
                "dataset_index_seconds": index_seconds,
                "random_iteration_seconds": iteration_seconds,
                "random_rows_per_second": len(index) / iteration_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
