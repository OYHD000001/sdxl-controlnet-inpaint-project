#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a JSONL metadata file into round-robin shards.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    shards = [[] for _ in range(args.num_shards)]
    for index, line in enumerate(lines):
        shards[index % args.num_shards].append(line)

    outputs = []
    for index, shard_lines in enumerate(shards):
        path = args.output_dir / f"shard_{index:02d}.jsonl"
        path.write_text("".join(f"{line}\n" for line in shard_lines), encoding="utf-8")
        outputs.append({"path": str(path.resolve()), "count": len(shard_lines)})

    print(json.dumps({"input": str(args.input.resolve()), "num_shards": args.num_shards, "shards": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
