from pathlib import Path


def test_rpp_single_launcher_uses_separate_init_and_reference_checkpoints() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "run_rpp_single_8gpu_refkl.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'INIT_CHECKPOINT="${INIT_CHECKPOINT:-${CHECKPOINT:-/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/' in content
    assert 'REFERENCE_POLICY_CHECKPOINT="${REFERENCE_POLICY_CHECKPOINT:-${INIT_CHECKPOINT}}"' in content
    assert 'CACHE_PATH="${CACHE_PATH:-/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train}"' in content
    assert 'CACHE_LOADER_MODE="${CACHE_LOADER_MODE:-legacy_cached_features}"' in content
    assert 'USE_CACHE_WITHOUT_DATASET="${USE_CACHE_WITHOUT_DATASET:-true}"' in content
    assert 'agent.checkpoint_path="\'${INIT_CHECKPOINT}\'"' in content
    assert 'agent.reference_policy_checkpoint="\'${REFERENCE_POLICY_CHECKPOINT}\'"' in content
    assert '+auto_resume_latest=false' in content


def test_rpp_single_launcher_threads_extra_overrides() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "run_rpp_single_8gpu_refkl.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"' in content
    assert 'read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES}"' in content
    assert '"${EXTRA_OVERRIDE_ARGS[@]}"' in content
