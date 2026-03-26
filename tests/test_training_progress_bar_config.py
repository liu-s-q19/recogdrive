from pathlib import Path


def test_default_training_exposes_progress_bar_refresh_rate() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "config"
        / "training"
        / "default_training.yaml"
    )

    content = config_path.read_text(encoding="utf-8")

    assert "progress_bar_refresh_rate:" in content
    assert "log_every_n_steps:" in content


def test_rl_training_entry_configures_tqdm_refresh_rate() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "run_training_recogdrive_rl.py"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'TQDMProgressBar(refresh_rate=cfg.trainer.progress_bar_refresh_rate)' in content


def test_sft_training_entry_configures_tqdm_refresh_rate() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "run_training_recogdrive_ema.py"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'TQDMProgressBar(refresh_rate=cfg.trainer.progress_bar_refresh_rate)' in content
