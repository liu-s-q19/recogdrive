from pathlib import Path

from omegaconf import OmegaConf


def test_v2_dataset_paths_require_explicit_runtime_version() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "common"
        / "default_dataset_paths.yaml"
    )

    assert config_path.exists()

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    assert "navsim_log_path" in cfg
    assert "original_sensor_path" in cfg
    assert "synthetic_sensor_path" in cfg
    assert "synthetic_scenes_path" in cfg
    assert "metric_cache_path" in cfg
    assert cfg["runtime_cache_version"] == "navsim_v2_recogdrive_1"


def test_dataset_caching_entrypoint_supports_v2_scene_loader_paths() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "run_dataset_caching.py"
    )

    content = script_path.read_text(encoding="utf-8")

    assert "original_sensor_path" in content
    assert "synthetic_sensor_path" in content
    assert "synthetic_scenes_path" in content
