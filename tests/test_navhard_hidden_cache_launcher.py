from pathlib import Path


def test_navhard_hidden_cache_launcher_uses_isolated_runtime_and_explicit_synthetic_overrides() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "cache_dataset"
        / "run_caching_recogdrive_hidden_state_navhard_two_stage.sh"
    )

    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"' in content
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"' in content
    assert 'TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navhard_two_stage}"' in content
    assert 'export NAVSIM_EXP_ROOT="$RUNTIME_ROOT/exp"' in content
    assert 'export NAVSIM_OUTPUT_ROOT="$RUNTIME_ROOT/outputs"' in content
    assert 'export TMPDIR="$RUNTIME_ROOT/tmp"' in content
    assert 'CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_navhard_two_stage}"' in content
    assert 'export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"' in content
    assert 'export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"' in content
    assert 'SYNTHETIC_SENSOR_PATH="${SYNTHETIC_SENSOR_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs}"' in content
    assert 'SYNTHETIC_SCENES_PATH="${SYNTHETIC_SCENES_PATH:-/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles}"' in content
    assert 'original_sensor_path="$OPENSCENE_DATA_ROOT/sensor_blobs/test_ini"' in content
    assert 'synthetic_sensor_path="$SYNTHETIC_SENSOR_PATH"' in content
    assert 'synthetic_scenes_path="$SYNTHETIC_SCENES_PATH"' in content
    assert 'agent.cache_hidden_state=True' in content
    assert 'agent.cache_mode=True' in content
