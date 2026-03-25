from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hydra.core.hydra_config import HydraConfig


V2_CACHE_SCHEMA_VERSION = "navsim_v2_recogdrive_1"
CACHE_META_FILE_NAME = "cache_meta.json"


def cache_meta_path(cache_root: Path) -> Path:
    return cache_root / CACHE_META_FILE_NAME


def load_cache_metadata(cache_root: Path) -> Optional[Dict[str, Any]]:
    meta_path = cache_meta_path(cache_root)
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_cache_metadata(
    cache_root: Path,
    train_test_split: str,
    scene_loader_mode: str,
    schema_version: str = V2_CACHE_SCHEMA_VERSION,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    meta_path = cache_meta_path(cache_root)
    with meta_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "schema_version": schema_version,
                "train_test_split": train_test_split,
                "scene_loader_mode": scene_loader_mode,
            },
            file_obj,
            indent=2,
            sort_keys=True,
        )
    return meta_path


def ensure_v2_cache_metadata(
    cache_root: Path,
    expected_schema_version: str = V2_CACHE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    metadata = load_cache_metadata(cache_root)
    if metadata is None:
        raise ValueError(f"Missing cache_meta.json in v2 cache root: {cache_root}")

    missing_keys = {"schema_version", "train_test_split", "scene_loader_mode"} - set(metadata.keys())
    if missing_keys:
        raise ValueError(
            f"cache_meta.json in {cache_root} is missing required keys: {sorted(missing_keys)}"
        )

    if metadata["schema_version"] != expected_schema_version:
        raise ValueError(
            f"Unsupported cache schema version {metadata['schema_version']} in {cache_root}; "
            f"expected {expected_schema_version}"
        )

    return metadata


def resolve_train_test_split_name(
    cfg: Any,
    hydra_choices: Optional[Mapping[str, Any]] = None,
    fallback: str = "unknown",
) -> str:
    if hydra_choices is None and HydraConfig.initialized():
        hydra_choices = HydraConfig.get().runtime.choices

    if hydra_choices is not None:
        selected_split = hydra_choices.get("train_test_split")
        if selected_split:
            return str(selected_split)

    cfg_get = getattr(cfg, "get", None)
    if callable(cfg_get):
        explicit_split_name = cfg_get("train_test_split_name")
        if explicit_split_name:
            return str(explicit_split_name)

    return fallback
