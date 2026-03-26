from pathlib import Path


def test_navhard_eval_launcher_forces_isolated_devkit_root() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh"
    )

    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"' in content
    assert 'export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"' in content
    assert 'NAVHARD_METRIC_CACHE_PATH="${NAVHARD_METRIC_CACHE_PATH:-/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733}"' in content
    assert 'METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVHARD_METRIC_CACHE_PATH}"' in content
    assert 'SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs}"' in content
    assert 'SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles}"' in content
    assert 'synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH"' in content
    assert 'synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"' in content
    assert 'output_dir="\'$OUTPUT_DIR\'"' in content
