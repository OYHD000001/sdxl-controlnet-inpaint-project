#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch training checkpoints and render intermediate preview composites.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def write_preview_config(config: dict[str, Any], checkpoint_dir: Path, output_dir: Path, metadata_path: Path) -> Path:
    preview_config = dict(config)
    preview_config["inference"] = dict(config["inference"])
    preview_config["inference"]["checkpoint_dir"] = str(checkpoint_dir)
    preview_config["inference"]["output_dir"] = str(output_dir)
    preview_config["inference"]["metadata_path"] = str(metadata_path)

    config_path = output_dir.parent / f"{output_dir.name}_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(preview_config, f, allow_unicode=True, sort_keys=False)
    return config_path


def run_command(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    project_root = ROOT
    output_dir = project_root / config["project"]["output_dir"]
    checkpoint_root = output_dir / "checkpoints"

    preview_cfg = config.get("preview", {})
    metadata_path = project_root / preview_cfg.get("metadata_path", config["data"]["train_metadata_path"])
    preview_root = ensure_dir(project_root / preview_cfg.get("output_dir", str(output_dir / "train_previews")))
    interval_steps = int(preview_cfg.get("every_steps", 1900))
    max_train_steps = int(config["training"]["max_train_steps"])
    max_wait_seconds = int(preview_cfg.get("max_wait_seconds", 7 * 24 * 3600))

    expected_steps = list(range(interval_steps, max_train_steps + 1, interval_steps))
    seen: set[int] = set()
    started_at = time.time()

    print(
        f"Watching checkpoints in {checkpoint_root} every {interval_steps} steps; preview metadata={metadata_path}",
        flush=True,
    )
    while len(seen) < len(expected_steps):
        if time.time() - started_at > max_wait_seconds:
            raise TimeoutError("Timed out waiting for training preview checkpoints.")

        for step in expected_steps:
            if step in seen:
                continue
            checkpoint_dir = checkpoint_root / f"step_{step:08d}"
            if not (checkpoint_dir / "train_state.json").exists():
                continue

            step_output_dir = ensure_dir(preview_root / f"step_{step:08d}" / "infer")
            step_config_path = write_preview_config(
                config=config,
                checkpoint_dir=checkpoint_dir,
                output_dir=step_output_dir,
                metadata_path=metadata_path,
            )
            run_command(
                [args.python, "-m", "src.training.validate", "--config", str(step_config_path), "--mode", "infer"],
                cwd=project_root,
            )
            run_command(
                [
                    args.python,
                    "scripts/make_infer_composites.py",
                    "--config",
                    str(step_config_path),
                    "--metadata-path",
                    str(metadata_path),
                    "--infer-dir",
                    str(step_output_dir),
                    "--output-dir",
                    str(preview_root / f"step_{step:08d}" / "composites"),
                ],
                cwd=project_root,
            )
            seen.add(step)
            print(f"Preview completed for step {step}.", flush=True)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
