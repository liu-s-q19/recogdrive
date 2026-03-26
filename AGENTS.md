# AGENTS.md

## Purpose

This repository is the isolated ReCogDrive NavSim v2 development line.

Use it to evolve the v2 codepath, configs, scripts, and validation flow without polluting the older `/data/liushiqi/recogdrive` workspace or its runtime outputs.

This file is a light guidance document for code agents. It does not prescribe a rigid step-by-step workflow, but it defines the current project boundaries and the defaults that should be treated as intentional.

## Working Scope

- Treat NavSim v2 development as the primary target.
- Prefer extending the v2 path over adding more one-off compatibility branches.
- Keep the legacy `navtest` path available as an explicit fallback until a v2 replacement is proven end-to-end.
- Do not write code or docs that assume the old repo root `/data/liushiqi/recogdrive` is the active workspace.

## Canonical Environment

Unless the current task explicitly says otherwise, assume these defaults:

- Code root: `/data/liushiqi/recogdrive-navsimv2`
- Current working folder: `/data/liushiqi/recogdrive-navsimv2`
- Runtime root: `/data/liushiqi/recogdrive-navsimv2-runtime`
- Conda env: `navsimv2-recogdrive`
- Python path: `/data/miniconda/envs/navsimv2-recogdrive/bin/python`
- Main evaluation split: `navhard_two_stage`
- Fallback evaluation split: `navtest`

Prefer environment variables over hardcoded absolute paths in scripts:

- `PROJECT_ROOT`
- `RUNTIME_ROOT`
- `NAVSIM_EXP_ROOT`
- `NAVSIM_OUTPUT_ROOT`
- `OPENSCENE_DATA_ROOT`
- `TMPDIR`
- `CONDA_ENV_NAME`

Runtime data defaults for the current machine:

- Standard dataset root: `/data/dataset/navsim`
- Default `navhard_two_stage` original sensor root: `/data/dataset/navsim/sensor_blobs/test_ini`
- Default `navhard_two_stage` synthetic sensor root: `/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`
- Default `navhard_two_stage` synthetic scene root: `/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles`
- Default `navhard_two_stage` metric cache root: `/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`

## Machine / SSH

- Current machine: `10.199.7.32` (`32`)
- The `data` directory is mutually accessible across these machines via SSH from the current host:
  - `10.199.7.32`
  - `10.199.7.33`
  - `10.199.7.190`
  - `10.199.7.191`
- Shared shell variables:

```bash
SERVER_IPS=("10.199.7.32" "10.199.7.33" "10.199.7.190" "10.199.7.191")
SSH_PORT=2289
SSH_USER="root"
```

- Standard SSH parameters:
  - `SSH_PORT=2289`
  - `SSH_USER=root`

## Long-Run Jobs (tmux)

- Long training, evaluation, caching, and submission jobs should run in background `tmux` sessions by default so SSH disconnects do not terminate them.
- Use clear, task-specific session names such as `train-rpp`, `eval-navhard`, or `cache-metric`.
- Redirect logs to files under the runtime root or task-specific output directories when possible.
- Before starting a new long job, check whether a matching session already exists to avoid duplicate runs.

Recommended template:

```bash
SESSION_NAME="eval-navhard"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /data/liushiqi/recogdrive-navsimv2 && \
   source /data/miniconda/etc/profile.d/conda.sh && \
   conda activate navsimv2-recogdrive && \
   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"

tmux ls
tmux attach -t "${SESSION_NAME}"
```

Validated full `navhard_two_stage` dedicated scorer command on this machine:

```bash
SESSION_NAME="eval-navhard-full"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /data/liushiqi/recogdrive-navsimv2 && \
   source /data/miniconda/etc/profile.d/conda.sh && \
   conda activate navsimv2-recogdrive && \
   export PROJECT_ROOT=/data/liushiqi/recogdrive-navsimv2 \
          RUNTIME_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime \
          NAVSIM_EXP_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime/exp \
          NAVSIM_OUTPUT_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime/outputs \
          TMPDIR=/data/liushiqi/recogdrive-navsimv2-runtime/tmp \
          OPENSCENE_DATA_ROOT=/data/dataset/navsim \
          NUPLAN_MAPS_ROOT=/data/dataset/navsim/maps \
          METRIC_CACHE_PATH=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733 \
          CHECKPOINT=/data/liushiqi/recogdrive/outputs/grpo/rpp1n8g_baseline_20260313_082058/lightning_logs/version_0/checkpoints/epoch=9-step=13300.ckpt \
          GPUS=8 && \
   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"

tmux ls
tmux attach -t "${SESSION_NAME}"
```

Successful output from that command currently lives under:

- `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/eval_navhard_two_stage`
- `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/eval_navhard_two_stage/2026.03.26.04.47.34.csv`
- `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/eval_navhard_two_stage/summary.json`

Recommended `navhard_two_stage` hidden-cache session:

```bash
SESSION_NAME="cache-navhard-hidden"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /data/liushiqi/recogdrive-navsimv2 && \
   source /data/miniconda/etc/profile.d/conda.sh && \
   conda activate navsimv2-recogdrive && \
   bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_navhard_two_stage.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

## Read First

Before making changes, read this file first:

- [AGENTS.md](/data/liushiqi/recogdrive-navsimv2/AGENTS.md)

Then read the files that define the current v2 line when they are relevant to the task:

- [docs/Train_Eval.md](/data/liushiqi/recogdrive-navsimv2/docs/Train_Eval.md)
- [task/navsimv2_isolated_migration_plan_2026-03-25.md](/data/liushiqi/recogdrive-navsimv2/task/navsimv2_isolated_migration_plan_2026-03-25.md)

If your task touches training, evaluation, caching, or dataset loading, also inspect the directly affected script and its config before changing anything.

## Current v2 Baseline

The repository currently reflects these v2-oriented constraints:

- Dataset/runtime paths are centralized in [navsim/planning/script/config/common/default_dataset_paths.yaml](/data/liushiqi/recogdrive-navsimv2/navsim/planning/script/config/common/default_dataset_paths.yaml).
- Runtime metric caches are expected to carry `cache_meta.json` metadata with schema version `navsim_v2_recogdrive_1`.
- The loader-side metadata helpers live in [navsim/common/cache_metadata.py](/data/liushiqi/recogdrive-navsimv2/navsim/common/cache_metadata.py).
- `MetricCacheLoader` can reject legacy caches when v2 metadata is required.
- Two-stage evaluation has a dedicated entrypoint in [navsim/planning/script/run_pdm_score_recogdrive_navhard.py](/data/liushiqi/recogdrive-navsimv2/navsim/planning/script/run_pdm_score_recogdrive_navhard.py).
- The default shell launcher for that path is [scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh](/data/liushiqi/recogdrive-navsimv2/scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh).

Do not silently undo these assumptions unless the task explicitly requires a design change.

Hidden-state cache policy for the isolated v2 line:

- `navhard_two_stage` hidden-state caches must be generated under `/data/liushiqi/recogdrive-navsimv2-runtime/exp`.
- Legacy `navtrain` hidden-state caches under `/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train` may still be reused for training-only fallback when `cache_loader_mode=legacy_cached_features`.
- Do not treat legacy `navtrain` hidden-state caches as valid `navhard_two_stage` evaluation caches.
- Old hidden-cache scripts under `scripts/cache_dataset/run_caching_recogdrive_hidden_state*.sh` are legacy entrypoints unless they explicitly target the isolated v2 runtime.

## Development Guidance

- Prefer the smallest change that keeps the v2 line coherent.
- Keep path handling centralized. If a new script needs roots or cache locations, thread them through config or env vars instead of re-hardcoding paths.
- Preserve the distinction between original sensor data and synthetic stage-two data.
- When touching cache generation or loading, update both the writer and the reader side if the schema contract changes.
- When touching Hydra configs, check that the split name, scene filter, and cache path assumptions still resolve together.
- If a task affects both docs and runnable scripts, update both in the same change.

## Guardrails

- Do not reuse old runtime directories under `/data/liushiqi/recogdrive/exp`, `/data/liushiqi/recogdrive/outputs`, or similar legacy roots for v2 work.
- Do not remove `navtest` fallback support unless the task explicitly replaces it and validates the replacement.
- Do not assume old metric caches are valid for v2. The metadata marker is part of the contract.
- Do not introduce new hardcoded references to the old repo root in launchers, configs, or docs.

## Verification

Prefer targeted verification that matches the area you touched.

For core v2 path/config/cache changes, start with:

```bash
pytest tests/test_v2_dataset_paths.py
pytest tests/test_v2_metric_cache_loader.py
pytest tests/test_navhard_two_stage_config.py
```

If you change a shell launcher, also run at least:

```bash
bash -n <script>
```

If you change a Hydra entrypoint, run the narrowest command that proves config resolution or argument wiring before claiming success.

## Documentation Expectations

When a change affects the v2 workflow, keep user-facing docs aligned:

- [AGENTS.md](/data/liushiqi/recogdrive-navsimv2/AGENTS.md) for agent-facing defaults and working rules
- [docs/Train_Eval.md](/data/liushiqi/recogdrive-navsimv2/docs/Train_Eval.md) for operational commands
- `task/*.md` notes when the change alters migration assumptions, runtime locations, or experiment procedures

## Preferred Task Framing For Agents

When reporting work, state clearly:

- whether the change affects training, evaluation, caching, dataset loading, or documentation
- whether it is v2-only or also preserves `navtest` fallback
- what verification was actually run
- any assumptions that were not verified end-to-end
