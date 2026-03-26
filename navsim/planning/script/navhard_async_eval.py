from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKPOINT_PATTERN = re.compile(r"epoch=(?P<epoch>\d+)-step=(?P<step>\d+)")


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def _parse_epoch_step(checkpoint_path: str) -> tuple[Optional[int], Optional[int]]:
    match = CHECKPOINT_PATTERN.search(Path(checkpoint_path).name)
    if not match:
        return None, None
    return int(match.group("epoch")), int(match.group("step"))


def _format_score(score: float, decimals: int) -> str:
    return f"{score:.{decimals}f}"


def _clear_ranked_symlinks(ranked_dir: Path) -> None:
    if not ranked_dir.exists():
        return
    for child in ranked_dir.iterdir():
        if child.is_symlink() or child.name.startswith(("rank", "best_navhard", "top2_navhard", "top3_navhard")):
            child.unlink(missing_ok=True)


def build_navhard_ranking(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked_entries: List[Dict[str, Any]] = []
    for item in evaluations:
        if item.get("status") != "succeeded":
            continue
        score = item.get("final_extended_pdm_score")
        if score is None:
            continue
        epoch, step = _parse_epoch_step(item["checkpoint_path"])
        ranked_entries.append(
            {
                **item,
                "epoch": epoch,
                "step": step,
            }
        )

    ranked_entries.sort(
        key=lambda item: (
            float(item["final_extended_pdm_score"]),
            item.get("epoch", -1) if item.get("epoch") is not None else -1,
            item.get("step", -1) if item.get("step") is not None else -1,
            item.get("completed_at", ""),
            item["checkpoint_path"],
        ),
        reverse=True,
    )

    for index, item in enumerate(ranked_entries, start=1):
        item["rank"] = index
    return ranked_entries


def _link_name(rank: int, score: float, checkpoint_path: str, score_decimals: int) -> str:
    checkpoint_name = Path(checkpoint_path).name
    return f"rank{rank}_score={_format_score(score, score_decimals)}_{checkpoint_name}"


def refresh_navhard_rankings(
    run_dir: Path,
    registry_path: Path,
    ranking_path: Path,
    top_k: int,
    score_decimals: int,
    keep_last: bool,
) -> Dict[str, Any]:
    registry = _read_json(registry_path, {"evaluations": []})
    ranking = build_navhard_ranking(registry.get("evaluations", []))

    ranked_dir = run_dir / "ranked_checkpoints"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    _clear_ranked_symlinks(ranked_dir)

    top_entries = ranking[:top_k]
    alias_names = ["best_navhard.ckpt", "top2_navhard.ckpt", "top3_navhard.ckpt"]

    for index, item in enumerate(top_entries, start=1):
        score = float(item["final_extended_pdm_score"])
        source_path = Path(item["checkpoint_path"]).resolve()
        link_path = ranked_dir / _link_name(index, score, item["checkpoint_path"], score_decimals)
        link_path.symlink_to(source_path)

        alias_name = alias_names[index - 1] if index <= len(alias_names) else f"top{index}_navhard.ckpt"
        alias_path = ranked_dir / alias_name
        alias_path.symlink_to(source_path)
        item["scored_checkpoint_link"] = str(link_path)
        item["alias_path"] = str(alias_path)

    last_checkpoint_path = None
    if keep_last:
        for candidate in sorted((run_dir / "lightning_logs").glob("version_*/checkpoints/last*.ckpt")):
            last_checkpoint_path = str(candidate.resolve())

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "all_ranked": ranking,
        "keep_last": keep_last,
        "last_checkpoint_path": last_checkpoint_path,
    }
    _write_json(ranking_path, payload)
    return payload


def register_checkpoint(
    registry_path: Path,
    checkpoint_path: str,
    eval_dir: str,
    session_name: str,
) -> bool:
    registry = _read_json(registry_path, {"evaluations": []})
    checkpoint_path = str(Path(checkpoint_path).resolve())

    for item in registry["evaluations"]:
        if item["checkpoint_path"] == checkpoint_path:
            return False

    epoch, step = _parse_epoch_step(checkpoint_path)
    registry["evaluations"].append(
        {
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "step": step,
            "eval_dir": str(Path(eval_dir).resolve()),
            "session_name": session_name,
            "status": "pending",
            "triggered_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(registry_path, registry)
    return True


def complete_evaluation(
    run_dir: Path,
    registry_path: Path,
    ranking_path: Path,
    checkpoint_path: str,
    summary_path: Optional[str],
    status: str,
    top_k: int,
    score_decimals: int,
    keep_last: bool,
) -> Dict[str, Any]:
    registry = _read_json(registry_path, {"evaluations": []})
    checkpoint_path = str(Path(checkpoint_path).resolve())
    score = None

    if summary_path:
        summary_candidate = Path(summary_path)
        if summary_candidate.exists():
            with summary_candidate.open("r", encoding="utf-8") as file_obj:
                summary = json.load(file_obj)
            score = summary.get("final_extended_pdm_score")

    for item in registry["evaluations"]:
        if item["checkpoint_path"] != checkpoint_path:
            continue
        item["status"] = status
        item["summary_path"] = str(Path(summary_path).resolve()) if summary_path else None
        item["completed_at"] = datetime.now(timezone.utc).isoformat()
        if score is not None:
            item["final_extended_pdm_score"] = float(score)
        break

    _write_json(registry_path, registry)
    return refresh_navhard_rankings(
        run_dir=run_dir,
        registry_path=registry_path,
        ranking_path=ranking_path,
        top_k=top_k,
        score_decimals=score_decimals,
        keep_last=keep_last,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage async navhard evaluation state")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register-checkpoint")
    register_parser.add_argument("--registry-path", required=True)
    register_parser.add_argument("--checkpoint-path", required=True)
    register_parser.add_argument("--eval-dir", required=True)
    register_parser.add_argument("--session-name", required=True)

    complete_parser = subparsers.add_parser("complete-eval")
    complete_parser.add_argument("--run-dir", required=True)
    complete_parser.add_argument("--registry-path", required=True)
    complete_parser.add_argument("--ranking-path", required=True)
    complete_parser.add_argument("--checkpoint-path", required=True)
    complete_parser.add_argument("--summary-path")
    complete_parser.add_argument("--status", required=True)
    complete_parser.add_argument("--top-k", type=int, default=3)
    complete_parser.add_argument("--score-decimals", type=int, default=6)
    complete_parser.add_argument("--keep-last", action="store_true")

    refresh_parser = subparsers.add_parser("refresh-ranking")
    refresh_parser.add_argument("--run-dir", required=True)
    refresh_parser.add_argument("--registry-path", required=True)
    refresh_parser.add_argument("--ranking-path", required=True)
    refresh_parser.add_argument("--top-k", type=int, default=3)
    refresh_parser.add_argument("--score-decimals", type=int, default=6)
    refresh_parser.add_argument("--keep-last", action="store_true")

    args = parser.parse_args()

    if args.command == "register-checkpoint":
        was_registered = register_checkpoint(
            registry_path=Path(args.registry_path),
            checkpoint_path=args.checkpoint_path,
            eval_dir=args.eval_dir,
            session_name=args.session_name,
        )
        print("registered" if was_registered else "exists")
        return

    if args.command == "complete-eval":
        payload = complete_evaluation(
            run_dir=Path(args.run_dir),
            registry_path=Path(args.registry_path),
            ranking_path=Path(args.ranking_path),
            checkpoint_path=args.checkpoint_path,
            summary_path=args.summary_path,
            status=args.status,
            top_k=args.top_k,
            score_decimals=args.score_decimals,
            keep_last=args.keep_last,
        )
        print(json.dumps(payload, sort_keys=True))
        return

    payload = refresh_navhard_rankings(
        run_dir=Path(args.run_dir),
        registry_path=Path(args.registry_path),
        ranking_path=Path(args.ranking_path),
        top_k=args.top_k,
        score_decimals=args.score_decimals,
        keep_last=args.keep_last,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
