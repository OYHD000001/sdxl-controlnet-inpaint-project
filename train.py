from __future__ import annotations

import argparse
from pathlib import Path

from src.training.train_controlnet_sdxl_inpaint import train_from_config
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical SDXL inpaint + pose ControlNet training entrypoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def validate_canonical_config(config: dict) -> None:
    if str(config.get("model", {}).get("base_mode", "inpaint")).lower() != "inpaint":
        raise AssertionError("Canonical training path requires model.base_mode = inpaint")
    if not bool(config.get("project", {}).get("canonical_pose_inpaint", False)):
        raise AssertionError("Canonical training path requires project.canonical_pose_inpaint = true")
    model_cfg = config["model"]
    controlnet_paths = model_cfg.get("controlnet_model_name_or_paths")
    if controlnet_paths is not None and len(controlnet_paths) != 1:
        raise AssertionError("Canonical path only allows a single pose ControlNet")
    if model_cfg.get("controlnet_model_name_or_path") in (None, "", "__from_unet__"):
        raise AssertionError("Canonical path requires an explicit pose ControlNet checkpoint")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_canonical_config(config)
    train_from_config(config)


if __name__ == "__main__":
    main()
