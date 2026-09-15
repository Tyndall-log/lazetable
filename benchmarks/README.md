# LazeTable 벤치마크

## SQLite DatasetIndex

실행:

```powershell
uv run python benchmarks/sqlite_dataset_index.py
```

2026-09-15, Windows, Python 3.13, 25,000개 합성 row 결과:

| 항목 | 결과 |
|---|---:|
| SQLite artifact 크기 | 4.06 MiB |
| SQLite artifact open | 54.21ms |
| train 22,500 row DatasetIndex 생성 | 51.47ms |
| random epoch 순회 | 27.32ms |
| random 순회 처리량 | 약 824K row/s |

이 측정은 metadata index만 대상으로 하며 payload 파일 읽기 및 모델 연산은 포함하지
않는다.
