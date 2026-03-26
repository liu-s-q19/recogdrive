from pathlib import Path

from navsim.planning.training.dataset import Dataset


class _DummyBuilder:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_unique_name(self) -> str:
        return self._name


def test_dataset_load_valid_caches_ignores_cache_metadata_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "log-a"
    token_dir = log_dir / "token-a"
    token_dir.mkdir(parents=True)
    (token_dir / "internvl_feature.gz").write_bytes(b"feature")
    (token_dir / "trajectory_target.gz").write_bytes(b"target")
    (tmp_path / "cache_meta.json").write_text(
        '{"schema_version":"navsim_v2_recogdrive_1","train_test_split":"navhard_two_stage","scene_loader_mode":"navsim_v2_scene_loader"}',
        encoding="utf-8",
    )

    valid_cache_paths = Dataset._load_valid_caches(
        tmp_path,
        [_DummyBuilder("internvl_feature")],
        [_DummyBuilder("trajectory_target")],
    )

    assert valid_cache_paths == {"token-a": token_dir}
