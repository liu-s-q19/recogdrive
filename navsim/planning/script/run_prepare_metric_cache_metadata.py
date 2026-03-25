from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from navsim.common.cache_metadata import (
    V2_CACHE_SCHEMA_VERSION,
    load_cache_metadata,
    write_cache_metadata,
)
from navsim.common.dataloader import MetricCacheLoader


REQUIRED_V2_CACHE_FIELDS = (
    "scene_type",
    "human_trajectory",
    "past_human_trajectory",
    "map_parameters",
)


def _load_indexed_cache_paths(cache_root: Path) -> List[Path]:
    metadata_dir = cache_root / "metadata"
    if not metadata_dir.is_dir():
        raise ValueError(f"Missing metadata/*.csv in external cache root: {cache_root}")

    csv_files = sorted(file for file in metadata_dir.iterdir() if file.suffix == ".csv")
    if not csv_files:
        raise ValueError(f"Missing metadata/*.csv in external cache root: {cache_root}")

    cache_paths: List[Path] = []
    for csv_file in csv_files:
        rows = csv_file.read_text(encoding="utf-8").splitlines()
        cache_paths.extend(Path(row.strip()) for row in rows[1:] if row.strip())

    if not cache_paths:
        raise ValueError(f"metadata/*.csv in {cache_root} does not contain any cache paths")

    return cache_paths


def _validate_metric_cache_object(metric_cache: Any, path: Path) -> None:
    missing_fields = [field_name for field_name in REQUIRED_V2_CACHE_FIELDS if not hasattr(metric_cache, field_name)]
    if missing_fields:
        raise ValueError(f"Metric cache object loaded from {path} is missing v2 fields: {missing_fields}")

    if getattr(metric_cache, "map_parameters", None) is None:
        raise ValueError(f"Metric cache object loaded from {path} is missing map_parameters")


def _resolve_sample_token(cache_paths: List[Path], sample_token: str | None) -> str:
    if sample_token is None:
        return cache_paths[0].parent.name

    for path in cache_paths:
        if path.parent.name == sample_token:
            return sample_token

    raise ValueError(f"Sample token {sample_token!r} was not found in metadata/*.csv")


def prepare_metric_cache_metadata(
    cache_root: Path,
    train_test_split: str,
    scene_loader_mode: str,
    runtime_cache_version: str = V2_CACHE_SCHEMA_VERSION,
    sample_token: str | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    cache_root = Path(cache_root)
    cache_paths = _load_indexed_cache_paths(cache_root)
    resolved_sample_token = _resolve_sample_token(cache_paths, sample_token)

    loader = MetricCacheLoader(cache_root, require_v2_metadata=False)
    metric_cache = loader.get_from_token(resolved_sample_token)
    sample_path = loader.metric_cache_paths[resolved_sample_token]
    _validate_metric_cache_object(metric_cache, sample_path)

    expected_metadata = {
        "schema_version": runtime_cache_version,
        "train_test_split": train_test_split,
        "scene_loader_mode": scene_loader_mode,
    }
    existing_metadata = load_cache_metadata(cache_root)
    if existing_metadata is not None and existing_metadata != expected_metadata and not overwrite:
        raise ValueError(
            "Found existing cache_meta.json with conflicting values; "
            "pass --overwrite to replace it"
        )

    write_cache_metadata(
        cache_root=cache_root,
        train_test_split=train_test_split,
        scene_loader_mode=scene_loader_mode,
        schema_version=runtime_cache_version,
    )

    return {
        "cache_root": str(cache_root),
        "cache_meta_path": str(cache_root / "cache_meta.json"),
        "schema_version": runtime_cache_version,
        "train_test_split": train_test_split,
        "scene_loader_mode": scene_loader_mode,
        "sample_token": resolved_sample_token,
        "sample_cache_path": str(sample_path),
        "indexed_row_count": len(cache_paths),
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adopt an external metric cache root by writing cache_meta.json.")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--train-test-split", required=True)
    parser.add_argument("--scene-loader-mode", required=True)
    parser.add_argument("--runtime-cache-version", default=V2_CACHE_SCHEMA_VERSION)
    parser.add_argument("--sample-token")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    summary = prepare_metric_cache_metadata(
        cache_root=args.cache_root,
        train_test_split=args.train_test_split,
        scene_loader_mode=args.scene_loader_mode,
        runtime_cache_version=args.runtime_cache_version,
        sample_token=args.sample_token,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
