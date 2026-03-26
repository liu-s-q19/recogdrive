from omegaconf import OmegaConf
from hydra import compose, initialize_config_module


def test_navhard_two_stage_hydra_config_resolves(monkeypatch) -> None:
    monkeypatch.setenv("OPENSCENE_DATA_ROOT", "/data/dataset/navsim")
    monkeypatch.setenv("NAVSIM_EXP_ROOT", "/tmp/navsim-exp")

    with initialize_config_module("navsim.planning.script.config.common", version_base=None):
        cfg = compose(config_name="default_common", overrides=["train_test_split=navhard_two_stage"])
        resolved = OmegaConf.to_container(cfg, resolve=True)

    assert cfg.train_test_split.data_split == "test"
    assert cfg.train_test_split.scene_filter.include_synthetic_scenes is True
    assert cfg.train_test_split.reactive_all_mapping
    assert resolved["original_sensor_path"] == "/data/dataset/navsim/sensor_blobs/test_ini"
