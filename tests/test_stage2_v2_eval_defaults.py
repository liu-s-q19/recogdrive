from pathlib import Path


def test_navtest_eval_launcher_defaults_to_v2_stage2_checkpoint() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "run_recogdrive_agent_pdm_score_evaluation_8b.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'CHECKPOINT="${CHECKPOINT:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_v2_8gpus_latest/' in content


def test_navhard_eval_launcher_defaults_to_v2_stage2_checkpoint() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluation"
        / "run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'CHECKPOINT="${CHECKPOINT:-${NAVSIM_OUTPUT_ROOT}/recogdrive_stage2_training_v2_8gpus_latest/' in content
