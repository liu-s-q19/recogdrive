import json
from pathlib import Path

import pytest


def test_default_training_exposes_checkpoint_config() -> None:
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

    assert "checkpoint:" in content
    assert "monitor:" in content
    assert "save_top_k:" in content
    assert "save_last:" in content
    assert "every_n_epochs:" in content
    assert "filename:" in content


def test_rl_training_entry_reads_checkpoint_config() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "navsim"
        / "planning"
        / "script"
        / "run_training_recogdrive_rl.py"
    )

    content = script_path.read_text(encoding="utf-8")

    assert 'checkpoint_cfg = cfg.get("checkpoint")' in content
    assert 'monitor=checkpoint_cfg.monitor' in content
    assert 'mode=checkpoint_cfg.mode' in content
    assert 'save_top_k=checkpoint_cfg.save_top_k' in content
    assert 'save_last=checkpoint_cfg.save_last' in content
    assert 'every_n_epochs=checkpoint_cfg.every_n_epochs' in content
    assert 'auto_insert_metric_name=False' in content


def test_real_navhard_ranking_creates_scored_aliases(tmp_path: Path) -> None:
    helpers = pytest.importorskip("navsim.planning.script.navhard_async_eval")

    ckpt_dir = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    ckpt_paths = [
        ckpt_dir / "epoch=09-step=13300.ckpt",
        ckpt_dir / "epoch=08-step=11800.ckpt",
        ckpt_dir / "epoch=07-step=10300.ckpt",
        ckpt_dir / "last.ckpt",
    ]
    for ckpt_path in ckpt_paths:
        ckpt_path.write_text("checkpoint", encoding="utf-8")

    registry_path = tmp_path / "navhard_eval_registry.json"
    ranking_path = tmp_path / "navhard_eval_ranking.json"

    registry = {
        "evaluations": [
            {
                "checkpoint_path": str(ckpt_paths[0]),
                "status": "succeeded",
                "summary_path": str(tmp_path / "eval1" / "summary.json"),
                "final_extended_pdm_score": 0.6285884975446979,
            },
            {
                "checkpoint_path": str(ckpt_paths[1]),
                "status": "succeeded",
                "summary_path": str(tmp_path / "eval2" / "summary.json"),
                "final_extended_pdm_score": 0.624103,
            },
            {
                "checkpoint_path": str(ckpt_paths[2]),
                "status": "succeeded",
                "summary_path": str(tmp_path / "eval3" / "summary.json"),
                "final_extended_pdm_score": 0.617442,
            },
            {
                "checkpoint_path": str(ckpt_paths[3]),
                "status": "pending",
            },
        ]
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    ranking = helpers.refresh_navhard_rankings(
        run_dir=tmp_path,
        registry_path=registry_path,
        ranking_path=ranking_path,
        top_k=3,
        score_decimals=6,
        keep_last=True,
    )

    ranked_dir = tmp_path / "ranked_checkpoints"
    assert [item["rank"] for item in ranking["all_ranked"]] == [1, 2, 3]
    assert "top_k" not in ranking
    assert ranked_dir.joinpath("best_navhard.ckpt").is_symlink()
    assert ranked_dir.joinpath("top2_navhard.ckpt").is_symlink()
    assert ranked_dir.joinpath("top3_navhard.ckpt").is_symlink()
    assert ranked_dir.joinpath("rank1_score=0.628588_epoch=09-step=13300.ckpt").is_symlink()
    assert ranked_dir.joinpath("rank2_score=0.624103_epoch=08-step=11800.ckpt").is_symlink()
    assert ranked_dir.joinpath("rank3_score=0.617442_epoch=07-step=10300.ckpt").is_symlink()
    assert ranking_path.exists()
    persisted = json.loads(ranking_path.read_text(encoding="utf-8"))
    assert "top_k" not in persisted


def test_real_navhard_ranking_breaks_score_ties_by_step(tmp_path: Path) -> None:
    helpers = pytest.importorskip("navsim.planning.script.navhard_async_eval")

    ckpt_dir = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    first = ckpt_dir / "epoch=03-step=300.ckpt"
    second = ckpt_dir / "epoch=04-step=400.ckpt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    ranking = helpers.build_navhard_ranking(
        [
            {"checkpoint_path": str(first), "status": "succeeded", "final_extended_pdm_score": 0.5},
            {"checkpoint_path": str(second), "status": "succeeded", "final_extended_pdm_score": 0.5},
        ]
    )

    assert ranking[0]["checkpoint_path"] == str(second)
    assert ranking[1]["checkpoint_path"] == str(first)
