# ReCogDrive Training and Evaluation

## NavSim v2 Isolated Workflow

For the isolated NavSim v2 migration line, use these defaults:

```bash
conda activate navsimv2-recogdrive
export PROJECT_ROOT=/data/liushiqi/recogdrive-navsimv2
export RUNTIME_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime
export NAVSIM_EXP_ROOT=$RUNTIME_ROOT/exp
export NAVSIM_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
export TMPDIR=$RUNTIME_ROOT/tmp
export OPENSCENE_DATA_ROOT=/data/dataset/navsim
```

- New v2 metric caches must be rebuilt under the isolated runtime root and now require `cache_meta.json` with schema `navsim_v2_recogdrive_1`.
- The default two-stage evaluation path is `scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh`.
- Runnable one-stage fallback for `navtest` remains available through `scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh` with `TRAIN_TEST_SPLIT=navtest`.
- Do not reuse the old `/data/liushiqi/recogdrive/exp`, `/data/liushiqi/recogdrive/outputs`, or `/data/liushiqi/recogdrive/exp/tmp` roots for NavSim v2 runs.
- Default `navhard_two_stage` synthetic assets on this machine are:
  - original sensor root: `/data/dataset/navsim/sensor_blobs/test_ini`
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles`
- Default `navhard_two_stage` metric cache root is:
  - `/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`
- Legacy `navtrain` hidden-state caches under `/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train` remain training-only fallback inputs for `legacy_cached_features`; do not use them as `navhard_two_stage` evaluation caches.

Canonical diffusion stage-2 v2 baseline flow:

```bash
SESSION_NAME=train-stage2-v2
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /data/liushiqi/recogdrive-navsimv2 && \
   source /data/miniconda/etc/profile.d/conda.sh && \
   conda activate navsimv2-recogdrive && \
   export PROJECT_ROOT=/data/liushiqi/recogdrive-navsimv2 \
          RUNTIME_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime \
          NAVSIM_EXP_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime/exp \
          NAVSIM_OUTPUT_ROOT=/data/liushiqi/recogdrive-navsimv2-runtime/outputs \
          TMPDIR=/data/liushiqi/recogdrive-navsimv2-runtime/tmp && \
   bash scripts/training/run_stage2_diffusion_sft_v2_8gpu.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

This stage-2 launcher intentionally reuses the legacy training cache at `/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train`, but all new logs and checkpoints land under `/data/liushiqi/recogdrive-navsimv2-runtime/outputs`. It also updates `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/recogdrive_stage2_training_v2_8gpus_latest` so the evaluation launchers can consume the newest v2 stage-2 checkpoint by default.

After stage-2 training completes, validate `navtest` and `navhard_two_stage` separately:

```bash
SESSION_NAME=eval-stage2-v2-navtest
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
          TRAIN_TEST_SPLIT=navtest \
          CACHE_LOADER_MODE=navsim_v2_scene_loader \
          CACHE_PATH_EVAL=/data/liushiqi/recogdrive-navsimv2-runtime/exp/recogdrive_agent_cache_dir_navtest_adopted \
          METRIC_CACHE_PATH=/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49 && \
   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"

SESSION_NAME=eval-stage2-v2-navhard
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
          CACHE_PATH_EVAL=/data/liushiqi/recogdrive-navsimv2-runtime/exp/recogdrive_agent_cache_dir_navhard_two_stage && \
   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

To generate the isolated `navhard_two_stage` hidden-state cache:

```bash
SESSION_NAME=cache-navhard-hidden
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /data/liushiqi/recogdrive-navsimv2 && \
   source /data/miniconda/etc/profile.d/conda.sh && \
   conda activate navsimv2-recogdrive && \
   bash scripts/cache_dataset/run_caching_recogdrive_hidden_state_navhard_two_stage.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

This writes hidden-state features to `/data/liushiqi/recogdrive-navsimv2-runtime/exp/recogdrive_agent_cache_dir_navhard_two_stage` by default and keeps the runtime output isolated from the legacy repo.

The current isolated `navhard_two_stage` path assumes:

- logs from `/data/dataset/navsim/navsim_logs/test`
- original sensor blobs from `/data/dataset/navsim/sensor_blobs/test_ini`
- synthetic camera assets from `/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`

This split is intentional for the current machine. The stage-two synthetic pickle set does not carry usable local lidar references, so the hidden-state cache path uses the ReCogDrive single-camera sensor config and does not require lidar loads.

If you need to reuse an external v2 metric cache root that already has `metadata/*.csv` but does not ship `cache_meta.json`, adopt it explicitly before running strict v2 evaluation:

```bash
/data/miniconda/envs/navsimv2-recogdrive/bin/python \
  /data/liushiqi/recogdrive-navsimv2/navsim/planning/script/run_prepare_metric_cache_metadata.py \
  --cache-root /data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733 \
  --train-test-split navhard_two_stage \
  --scene-loader-mode navsim_v2_scene_loader \
  --runtime-cache-version navsim_v2_recogdrive_1
```

Then point the existing strict navhard evaluation entrypoint at that adopted root:

```bash
METRIC_CACHE_PATH=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733 \
bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_navhard_two_stage.sh
```

This is an explicit external-cache adoption flow, not a relaxation of the default v2 cache contract. Rebuilding caches under the isolated runtime root is still the preferred path.

For the `navtest` fallback smoke in the isolated v2 line, keep the cache roots explicit and do not rely on inherited tmux environment:

```bash
SESSION_NAME=eval-navtest-fallback-v2
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
          TRAIN_TEST_SPLIT=navtest \
          CACHE_LOADER_MODE=navsim_v2_scene_loader \
          CACHE_PATH_EVAL=/data/liushiqi/recogdrive-navsimv2-runtime/exp/recogdrive_agent_cache_dir_navtest_adopted \
          METRIC_CACHE_PATH=/data/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49 \
          ENTRYPOINT=/data/liushiqi/recogdrive-navsimv2/navsim/planning/script/run_pdm_score_recogdrive.py && \
   bash scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_8b.sh"
```

The fallback launcher now accepts env-driven `METRIC_CACHE_PATH`, `CACHE_LOADER_MODE`, `NAVSIM_DEVKIT_ROOT`, `NUPLAN_MAPS_ROOT`, `OPENSCENE_DATA_ROOT`, and `PYTHONPATH`, which is required for this smoke to stay on the isolated worktree.

Short isolated training smokes validated on `2026-03-26 UTC`:

```bash
# IL smoke: direct single-GPU, cached navtrain fallback
torchrun --standalone --nproc_per_node=1 \
  /data/liushiqi/recogdrive-navsimv2/navsim/planning/script/run_training_recogdrive.py \
  agent=recogdrive_agent \
  agent.vlm_path=/data/liushiqi/recogdrive-navsimv2/ckpt/ReCogDrive-VLM-8B \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.vlm_type=internvl \
  agent.dit_type=small \
  train_test_split=navtrain \
  cache_loader_mode=legacy_cached_features \
  cache_path=/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  trainer.params.devices=1 \
  trainer.params.num_nodes=1 \
  trainer.params.max_epochs=1 \
  trainer.params.limit_train_batches=1 \
  trainer.params.limit_val_batches=1

# RL smoke: isolated tiny subset through run_rpp_single_8gpu_refkl.sh
GPUS_PER_NODE=1 MAX_EPOCHS=1 DATALOADER_BATCH_SIZE=1 DATALOADER_NUM_WORKERS=1 \
CACHE_PATH=/data/liushiqi/recogdrive-navsimv2-runtime/exp/recogdrive_agent_cache_dir_train_smoke_subset \
METRIC_CACHE_PATH=/data/liushiqi/recogdrive-navsimv2-runtime/exp/metric_cache_train_smoke_subset \
CACHE_LOADER_MODE=legacy_cached_features USE_CACHE_WITHOUT_DATASET=true \
bash scripts/training/run_rpp_single_8gpu_refkl.sh
```

For v2 RL launchers, checkpoint roles are now explicit:

- `INIT_CHECKPOINT`: initializes the trainable student policy. Default is the v1 stage-2 SFT EMA family under `/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus`, preferring `last-EMA.ckpt` and falling back to `last.ckpt`.
- `REFERENCE_POLICY_CHECKPOINT`: frozen teacher/reference policy used by both reference-KL and BC regularization. Default is `INIT_CHECKPOINT`.
- `CHECKPOINT` in evaluation scripts: evaluation-only model input. It does not inherit RL `REFERENCE_POLICY_CHECKPOINT`.

This keeps the first v2 RL development phase compatible with the v1 regularization behavior while allowing later teacher-policy decoupling without changing the launcher interface.

Validated `10.199.7.33` single-node 8-GPU v2 RL bring-up on `2026-03-26 UTC`:

```bash
SESSION_NAME=train-rpp-v2-33
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
          CONDA_ENV_NAME=navsimv2-recogdrive \
          TRAIN_TEST_SPLIT=navtrain \
          CACHE_LOADER_MODE=legacy_cached_features \
          USE_CACHE_WITHOUT_DATASET=true \
          CACHE_PATH=/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train \
          METRIC_CACHE_PATH=/data/liushiqi/recogdrive/exp/metric_cache_train \
          INIT_CHECKPOINT=/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints \
          OUTPUT_DIR=/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_$(date +%Y%m%d_%H%M%S) \
          GPUS_PER_NODE=8 && \
   bash scripts/training/run_rpp_single_8gpu_refkl.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

Notes for that command:

- `METRIC_CACHE_PATH` must use the training cache root `/data/liushiqi/recogdrive/exp/metric_cache_train` for `train_test_split=navtrain`.
- The outer tmux log only records the launcher banner and resolved paths:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/train-rpp-v2-33.log`
- The real training logs are written inside the run output directory:
  - `train_rank0_<timestamp>.log`
  - `run_training_recogdrive_rl.log`
  - `lightning_logs/version_0/events.out.tfevents.*`
- RL launchers now default to epoch-level checkpoint retention for real async eval:
  - `CKPT_MONITOR=null`
  - `CKPT_SAVE_TOP_K=-1`
  - `CKPT_SAVE_LAST=true`
  - `CKPT_FILENAME=epoch={epoch:02d}-step={step}`
- RL launchers also start an async navhard watcher by default:
  - `REAL_EVAL_ENABLED=1`
  - `REAL_EVAL_SPLIT=navhard_two_stage`
  - `REAL_EVAL_ASYNC_MODE=tmux`
  - `REAL_EVAL_TOP_K=3`
  - `REAL_EVAL_SCORE_DECIMALS=6`

Real navhard async eval outputs for each RL run now live under the training run directory:

- registry:
  - `<RUN_DIR>/navhard_eval_registry.json`
- ranking manifest:
  - `<RUN_DIR>/navhard_eval_ranking.json`
- scored aliases:
  - `<RUN_DIR>/ranked_checkpoints/best_navhard.ckpt`
  - `<RUN_DIR>/ranked_checkpoints/top2_navhard.ckpt`
  - `<RUN_DIR>/ranked_checkpoints/top3_navhard.ckpt`
  - `<RUN_DIR>/ranked_checkpoints/rank1_score=<score>_epoch=<epoch>-step=<step>.ckpt`

The watcher polls new epoch checkpoints and launches one tmux eval session per unseen checkpoint through [watch_epoch9_and_eval.sh](/data/liushiqi/recogdrive-navsimv2/scripts/evaluation/watch_epoch9_and_eval.sh). Despite the historical filename, this script is now the generic RL async navhard watcher.

To inspect current async eval state for a run:

```bash
cat <RUN_DIR>/navhard_eval_registry.json
cat <RUN_DIR>/navhard_eval_ranking.json
ls -l <RUN_DIR>/ranked_checkpoints
tmux ls | grep eval-navhard-rpp
```

The ranking manifest is now the source of truth for RL best-model selection. It persists only `all_ranked`; the watcher still uses `REAL_EVAL_TOP_K` internally when creating `best_navhard.ckpt`, `top2_navhard.ckpt`, and other ranked symlinks, but JSON consumers should slice `all_ranked[:K]` themselves. `navtest` evaluation remains available as a manual fallback, but it is no longer the RL primary selection metric.

Observed progress from the validated `33` run:

- output dir:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_20260326_080415`
- real rank0 log:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_20260326_080415/train_rank0_20260326_080417.log`
- training script log:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_20260326_080415/run_training_recogdrive_rl.log`
- verified behavior:
  - `torchrun --nproc_per_node=8` started successfully
  - 8 worker processes entered `run_training_recogdrive_rl.py`
  - cached-feature training path loaded:
    - `Num training samples: 85109`
    - `Num validation samples: 18179`
  - Lightning entered live optimization and advanced through `Epoch 0`
  - observed progress reached roughly `66/1330` training steps before manual stop
  - live metrics were emitted in rank0 log:
    - `train/reward_step`
    - `train/policy_loss_step`
    - `train/bc_loss_step`

The `2026-03-26` `33` run was stopped manually, so the shutdown trace ends with `SignalException ... got signal: 1`. That stop was operator-triggered and does not indicate a training-path failure.

Validated `10.199.7.33` full RFT/RL navhard async-eval run on `2026-03-26 UTC`:

```bash
SESSION_NAME=train-rpp-v2-33-navhard-full
RUN_DIR=/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_$(date +%Y%m%d_%H%M%S)
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
          CONDA_ENV_NAME=navsimv2-recogdrive \
          TRAIN_TEST_SPLIT=navtrain \
          CACHE_LOADER_MODE=legacy_cached_features \
          USE_CACHE_WITHOUT_DATASET=true \
          CACHE_PATH=/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train \
          METRIC_CACHE_PATH=/data/liushiqi/recogdrive-navsimv2-runtime/exp/metric_cache_train \
          NAVHARD_METRIC_CACHE_PATH=/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733 \
          INIT_CHECKPOINT=/data/liushiqi/recogdrive/outputs/recogdrive_stage2_training_ema_multinode_8gpus/lightning_logs/version_10/checkpoints \
          OUTPUT_DIR=${RUN_DIR} \
          GPUS_PER_NODE=8 \
          REAL_EVAL_ENABLED=1 \
          REAL_EVAL_SPLIT=navhard_two_stage \
          REAL_EVAL_ASYNC_MODE=tmux \
          REAL_EVAL_TOP_K=3 \
          REAL_EVAL_SCORE_DECIMALS=6 \
          REAL_EVAL_GPUS=8 \
          REAL_EVAL_SESSION_PREFIX=eval-navhard-rpp-33full && \
   bash scripts/training/run_rpp_single_8gpu_refkl.sh \
   > /data/liushiqi/recogdrive-navsimv2-runtime/outputs/${SESSION_NAME}.log 2>&1"
```

Operational notes for that full run:

- Training and real `navhard_two_stage` eval both run on `10.199.7.33`.
- After each epoch checkpoint is written, the watcher starts a separate tmux eval session on `33`.
- The training process does not wait for eval completion:
  - `epoch N+1` training overlaps with real navhard evaluation for `epoch N`.
- The runtime-verified full-run directory is:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632`
- The active watcher session for that run is:
  - `eval-navhard-rpp-33full-rpp_v2_33_navhard_full_20260326_112632`

Runtime-verified artifacts from that full run:

- checkpoints:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/lightning_logs/version_0/checkpoints/epoch=00-step=1330.ckpt`
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/lightning_logs/version_0/checkpoints/epoch=01-step=2660.ckpt`
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/lightning_logs/version_0/checkpoints/last.ckpt`
- async eval registry:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/navhard_eval_registry.json`
- async eval ranking:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/navhard_eval_ranking.json`
- scored checkpoint aliases:
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/ranked_checkpoints/best_navhard.ckpt`
  - `/data/liushiqi/recogdrive-navsimv2-runtime/outputs/grpo/rpp_v2_33_navhard_full_20260326_112632/ranked_checkpoints/rank1_score=0.500703_epoch=00-step=1330.ckpt`

Current verified scoring status for that run:

- `epoch=00-step=1330.ckpt` async navhard eval succeeded with:
  - `final_extended_pdm_score = 0.5007030081068768`
- `epoch=01-step=2660.ckpt` async navhard eval was auto-triggered and is running normally.

One recurring log pattern in this full run is:

```text
OSError: [Errno 39] Directory not empty: '/data/liushiqi/recogdrive-navsimv2-runtime/tmp/pymp-*'
```

This is a non-fatal Python multiprocessing temp-directory cleanup race during dataloader worker teardown. In the verified `33` full run it does not stop training, does not block checkpointing, and does not block real navhard async evaluation.

## Stage 1: Vision-Language Models Driving Pretraining

First, you need to download **13 QA datasets** (e.g., *DriveLM*, *LingoQA*, etc.) as mentioned in the paper.  
Due to dataset privacy policies, we are currently unable to release the JSON files. These files may be released later if permission is granted by the dataset authors. Once obtained, you should configure the corresponding JSON files under `./internvl_chat/shell/data_info`.

You can also generate the **ReCogDrive dataset on NAVSIM** following the steps below:

```bash
cd ./scripts
sh generate_dataset/generate_internvl_dataset.sh              # trajectory dataset
sh generate_dataset/generate_internvl_dataset_pipeline.sh     # auto-labeled dataset with pipeline
```
Note: Before running the pipeline script, you need to deploy the corresponding VLM using vllm or Sglang for automatic generation.

Next, download the **InternVL pretrained weights** from HuggingFace:  
👉 [InternVL3-2B Weights](https://huggingface.co/OpenGVLab/InternVL3-2B)
👉 [InternVL3-8B Weights](https://huggingface.co/OpenGVLab/InternVL3-8B)

After downloading, go to `./internvl_chat/shell/internvl3.0/2nd_finetune` and configure the training script.  
You can launch the pretraining process with the following commands:

```bash
cd /path/to/internvl_chat
sh ./shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
```


## Stage 2: Diffusion Planner Imitation Learning

You can download our pretrained **ReCogDrive VLM** from [ReCogDrive VLM](https://huggingface.co/collections/owl10/recogdrive-68bafa143de172bab8de5752).  

For the diffusion planner training, the first step is to **cache datasets for faster training**.  
Since DiT training converges relatively slowly, training VLM and DiT jointly can be very time-consuming. To accelerate, we cache the hidden states output by the VLM, which enables much faster training.  
> ⚠️ Note: Caching requires approximately **1–2 TB of disk space**. We are also working on faster training methods.  

We also provide the option to skip caching hidden states and directly train VLM + DiT together, though this will be slower. We recommend using ReCogDrive-2B for training for better efficiency.

### Step 1: Cache hidden states
```bash
# cache dataset for training
sh cache_dataset/run_caching_recogdrive_hidden_state.sh

# cache dataset for evaluation
sh cache_dataset/run_caching_recogdrive_hidden_state_eval.sh

# cache dataset for evaluation without caching hidden state
sh cache_dataset/run_caching_recogdrive_no_hidden_state_eval.sh

```

### Step 2: Configure and run training

Configure the script `training/run_recogdrive_train_multi_node_2b.sh` and then start training:

```bash
sh training/run_recogdrive_train_multi_node_2b.sh
```

You can also enable **EMA (Exponential Moving Average)** during training for faster convergence. Note that this may lead to very slight performance degradation.

```bash
sh training/run_recogdrive_train_multi_node_ema_2b.sh
```

### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh

or

sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_no_hidden_state.sh

```

This will evaluate your trained agent using **PDM scores** on the navtest.




## Stage 3: Diffusion Planner Reinforcement Learning Training

In this stage, we perform **reinforcement learning (RL) training** on the Diffusion Planner  to further improve planning performance.

### Step 1: Metric Caching

First, you need to cache metrics for the training and test sets, which will be used for evaluation during RL training.

> ⚠️ **Note:** As mentioned in [Issue #10](https://github.com/xiaomi-research/recogdrive/issues/10#issuecomment-3344730681), you **must use NumPy version 1.26.4 or above** to avoid potential errors during metric caching.

```bash
# cache metrics for navtrain
sh cache_dataset/run_metric_caching_train.sh

# cache metrics for navtest
sh cache_dataset/run_metric_caching.sh
```


### Step 2: Configure and Launch RL Training

After caching metrics, configure the RL training script and launch training:

```bash
# Example path to the RL training script
sh training/run_recogdrive_train_multi_node_rl_2b.sh
```

Before running, modify the script parameters as needed  according to your hardware and training requirements. This command will start RL training immediately after configuration.


### Step 3: Configure and Run Evaluation

After training is complete, you can configure the evaluation script and launch evaluation:

```bash
sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh

or

sh evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_no_hidden_state.sh

```
This will evaluate your trained agent using **PDM scores** on the navtest.
