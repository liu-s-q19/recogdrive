from pathlib import Path


def test_watcher_script_supports_real_navhard_async_eval() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "watch_epoch9_and_eval.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"' in content
    assert 'REGISTRY_PATH="${REGISTRY_PATH:-${RUN_DIR}/navhard_eval_registry.json}"' in content
    assert 'RANKING_PATH="${RANKING_PATH:-${RUN_DIR}/navhard_eval_ranking.json}"' in content
    assert 'REAL_EVAL_TOP_K="${REAL_EVAL_TOP_K:-3}"' in content
    assert 'REAL_EVAL_SCORE_DECIMALS="${REAL_EVAL_SCORE_DECIMALS:-6}"' in content
    assert 'REAL_EVAL_GPUS="${REAL_EVAL_GPUS:-8}"' in content
    assert 'NAVHARD_METRIC_CACHE_PATH="${NAVHARD_METRIC_CACHE_PATH:-/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733}"' in content
    assert "GPUS='${REAL_EVAL_GPUS}'" in content
    assert "METRIC_CACHE_PATH='${NAVHARD_METRIC_CACHE_PATH}'" in content
    assert "OPENSCENE_DATA_ROOT" in content
    assert "NUPLAN_MAPS_ROOT" in content
    assert 'tmux new-session -d -s "${session_name}"' in content
    assert "navsim.planning.script.navhard_async_eval" in content
