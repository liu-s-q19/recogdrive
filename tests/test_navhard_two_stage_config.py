from hydra import compose, initialize_config_module


def test_navhard_two_stage_hydra_config_resolves() -> None:
    with initialize_config_module("navsim.planning.script.config.common", version_base=None):
        cfg = compose(config_name="default_common", overrides=["train_test_split=navhard_two_stage"])

    assert cfg.train_test_split.data_split == "test"
    assert cfg.train_test_split.scene_filter.include_synthetic_scenes is True
    assert cfg.train_test_split.reactive_all_mapping
