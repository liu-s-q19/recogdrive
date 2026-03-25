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
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/sensor_blobs`
  - `/readOnly/df_l2.9/navsim/navhard_two_stage/synthetic_scene_pickles`
- Default `navhard_two_stage` metric cache root is:
  - `/data/dataset/navsim/metric_cache_v2/navhard_two_stage_full_2026-03-09_03-37-22_n733`
- Legacy `navtrain` hidden-state caches under `/data/liushiqi/recogdrive/exp/recogdrive_agent_cache_dir_train` remain training-only fallback inputs for `legacy_cached_features`; do not use them as `navhard_two_stage` evaluation caches.

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
