#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch completion progress for external mannequin inference outputs.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=30)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def generated_path(record: dict, output_root: Path) -> Path:
    subdir = record.get("output_subdir")
    output_name = record.get("output_name") or Path(record["source_image"]).stem
    if subdir:
        return output_root / subdir / f"{output_name}_generated.png"
    return output_root / f"{output_name}_generated.png"


def summarize(records: list[dict], output_root: Path) -> tuple[int, int, Counter, Counter]:
    total = len(records)
    done = 0
    total_by_group: Counter = Counter()
    done_by_group: Counter = Counter()
    for record in records:
        group = record.get("output_subdir") or "root"
        total_by_group[group] += 1
        if generated_path(record, output_root).exists():
            done += 1
            done_by_group[group] += 1
    return total, done, total_by_group, done_by_group


def main() -> None:
    args = parse_args()
    records = load_records(args.metadata)
    while True:
        total, done, total_by_group, done_by_group = summarize(records, args.output_root)
        print("=" * 80, flush=True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print(f"overall: {done}/{total} ({(100.0 * done / total) if total else 0:.2f}%)", flush=True)
        for group in sorted(total_by_group):
            group_done = done_by_group[group]
            group_total = total_by_group[group]
            print(f"{group}: {group_done}/{group_total} ({(100.0 * group_done / group_total) if group_total else 0:.2f}%)", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
