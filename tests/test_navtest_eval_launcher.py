from pathlib import Path


def test_navtest_eval_launcher_exposes_v2_cache_overrides() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "run_recogdrive_agent_pdm_score_evaluation_8b.sh"
    )

    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$PROJECT_ROOT/dataset/navsim/maps}"' in content
    assert 'export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_ROOT}"' in content
    assert 'export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$PROJECT_ROOT/dataset/navsim}"' in content
    assert 'export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"' in content
    assert 'METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"' in content
    assert 'CACHE_LOADER_MODE="${CACHE_LOADER_MODE:-legacy_cached_features}"' in content
    assert 'metric_cache_path=$METRIC_CACHE_PATH' in content
