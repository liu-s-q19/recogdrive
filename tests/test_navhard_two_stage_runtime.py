from types import SimpleNamespace

from omegaconf import OmegaConf

from navsim.common.cache_metadata import resolve_train_test_split_name
from navsim.planning.script.run_pdm_score_recogdrive_navhard import (
    _configure_stage_filters,
    _resolve_stage_two_tokens,
)


def test_resolve_train_test_split_name_prefers_hydra_choice() -> None:
    cfg = OmegaConf.create(
        {
            "train_test_split": {
                "data_split": "test",
            }
        }
    )

    resolved = resolve_train_test_split_name(
        cfg,
        hydra_choices={"train_test_split": "navhard_two_stage"},
    )

    assert resolved == "navhard_two_stage"


def test_configure_stage_filters_routes_stage_two_tokens_to_synthetic_filter() -> None:
    stage_one_filter, stage_two_filter = _configure_stage_filters(
        OmegaConf.create(
            {
                "tokens": ["original-a", "original-b"],
                "synthetic_scene_tokens": None,
            }
        ),
        stage_one_tokens=["original-a"],
        stage_two_tokens=["synthetic-a"],
    )

    assert stage_one_filter.tokens == ["original-a"]
    assert stage_one_filter.synthetic_scene_tokens is None
    assert stage_two_filter.tokens is None
    assert stage_two_filter.synthetic_scene_tokens == ["synthetic-a"]


def test_resolve_stage_two_tokens_preserves_explicit_empty_reactive_subset() -> None:
    loader = SimpleNamespace(
        reactive_tokens_stage_two=[],
        synthetic_scenes={"synthetic-a": object(), "synthetic-b": object()},
    )

    resolved = _resolve_stage_two_tokens(loader, metric_cache_tokens=["synthetic-a", "synthetic-b"])

    assert resolved == []
