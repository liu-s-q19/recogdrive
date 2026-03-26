from pathlib import Path


def test_stage2_v2_launcher_uses_isolated_runtime_and_legacy_train_cache() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "run_stage2_diffusion_sft_v2_8gpu.sh"
    )

    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'RUNTIME_ROOT="${RUNTIME_ROOT:-/data/liushiqi/recogdrive-navsimv2-runtime}"' in content
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-navsimv2-recogdrive}"' in content
    assert 'export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"' in content
    assert 'VLM_PATH="${VLM_PATH:-${PROJECT_ROOT}/ckpt/ReCogDrive-VLM-8B}"' in content
    assert 'CACHE_PATH="${CACHE_PATH:-/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train}"' in content
    assert 'TRAIN_ENTRY="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_ema.py"' in content
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_v2_8gpus_' in content
    assert 'use_cache_without_dataset=True' in content
    assert 'force_cache_computation=False' in content
    assert 'agent.grpo=False' in content
    assert 'agent.cache_hidden_state=True' in content
    assert 'agent.vlm_type="internvl"' in content
    assert 'agent.dit_type="small"' in content
    assert 'agent.sampling_method="ddim"' in content
    assert 'train_test_split="${TRAIN_TEST_SPLIT}"' in content


def test_stage2_v2_launcher_exposes_smoke_overrides() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "run_stage2_diffusion_sft_v2_8gpu.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'MAX_EPOCHS="${MAX_EPOCHS:-100}"' in content
    assert 'BATCH_SIZE="${BATCH_SIZE:-16}"' in content
    assert 'NUM_WORKERS="${NUM_WORKERS:-8}"' in content
    assert 'PIN_MEMORY="${PIN_MEMORY:-false}"' in content
    assert 'LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"' in content
    assert 'LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"' in content
    assert 'MASTER_PORT="${MASTER_PORT:-63689}"' in content
    assert 'LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_stage2_v2_${TS}.log}"' in content
