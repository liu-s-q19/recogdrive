import time
import torch
import os
import numpy as np
from navsim.common.dataclasses import TrajectorySampling
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from transformers.feature_extraction_utils import BatchFeature

# ---------------- 配置区域 ----------------
CHECKPOINT_PATH = "/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/exp/recogdrive_stage3_rl_training_16gpus_bs8/lightning_logs/version_0/checkpoints/epoch=9-step=6650.ckpt"
VLM_PATH = "/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/ckpt/ReCogDrive-VLM-8B"
DEVICE_ID = 0
# ------------------------------------------

def main():
    device = torch.device(f"cuda:{DEVICE_ID}")
    print(f"🚀 Starting Benchmark on {torch.cuda.get_device_name(device)}...")

    print("Loading Model...")
    traj_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)

    agent = ReCogDriveAgent(
        trajectory_sampling=traj_sampling,
        checkpoint_path=CHECKPOINT_PATH,
        vlm_path=VLM_PATH,
        cam_type='single',
        vlm_type='internvl',
        dit_type='small',
        sampling_method='ddim',
        cache_mode=False,         
        cache_hidden_state=False, 
        vlm_size='large',        
        grpo=False,
    ).to(device)
    
    agent.initialize()
    agent.eval()

    # 获取 Diffusion Head 的数据类型
    model_dtype = next(agent.action_head.parameters()).dtype
    print(f"Diffusion Model Dtype: {model_dtype}")

    # ---------------- 2. 构造全套模拟输入 ----------------
    # A. VLM 输入
    dummy_images = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16).to(device)
    dummy_questions = ["<image>\nPredict trajectory."] 
    dummy_num_patches_list = [1]

    # B. Diffusion 输入数据 (根据报错反推的正确维度)
    # status_feature 必须是 8 维
    raw_status = torch.randn(1, 8).to(device) 
    # history trajectory 通常是 4步 * 3维 = 12 维
    raw_his_traj = torch.randn(1, 12).to(device)
    # state 是上面两者的拼接: 8 + 12 = 20 维
    raw_state = torch.randn(1, 20).to(device) 

    # 构造 BatchFeature 并转换类型
    dummy_action_inputs = BatchFeature({
        "state": raw_state.to(dtype=model_dtype), 
        "his_traj": raw_his_traj.to(dtype=model_dtype),
        "status_feature": raw_status.to(dtype=model_dtype)
    })

    # ---------------- 3. 预热 (Warmup) ----------------
    print("\nStarting Warmup...")
    warmup_hidden_state = None
    with torch.no_grad():
        for _ in range(3):
            out = agent.backbone(dummy_images, dummy_questions, dummy_num_patches_list)
            warmup_hidden_state = out.hidden_states[-1]
            
            _ = agent.action_head.get_action(
                warmup_hidden_state.to(dtype=model_dtype), 
                dummy_action_inputs
            )
    print("Warmup Done.")

    # ---------------- 4. 正式测速 ----------------
    loops = 50
    
    # === Test A: VLM Backbone ===
    start_vlm = torch.cuda.Event(enable_timing=True)
    end_vlm = torch.cuda.Event(enable_timing=True)
    
    start_vlm.record()
    with torch.no_grad():
        for _ in range(loops):
            _ = agent.backbone(dummy_images, dummy_questions, dummy_num_patches_list)
    end_vlm.record()
    torch.cuda.synchronize()
    avg_vlm = start_vlm.elapsed_time(end_vlm) / loops

    # === Test B: Diffusion Planner ===
    ready_hidden_state = warmup_hidden_state.to(dtype=model_dtype)

    start_plan = torch.cuda.Event(enable_timing=True)
    end_plan = torch.cuda.Event(enable_timing=True)
    
    start_plan.record()
    with torch.no_grad():
        for _ in range(loops):
            _ = agent.action_head.get_action(ready_hidden_state, dummy_action_inputs)
    end_plan.record()
    torch.cuda.synchronize()
    avg_plan = start_plan.elapsed_time(end_plan) / loops

    # ---------------- 5. 打印报告 ----------------
    total_time = avg_vlm + avg_plan
    fps = 1000 / total_time

    print("\n" + "="*50)
    print(f"📊 Inference Benchmark Report (H20 GPU)")
    print("="*50)
    print(f"1. VLM Encoding (InternVL-8B): {avg_vlm:.2f} ms")
    print(f"2. Diffusion Planning (DDIM):  {avg_plan:.2f} ms")
    print("-" * 50)
    print(f"🏆 Total Latency:              {total_time:.2f} ms")
    print(f"🚀 Est. Throughput:            {fps:.2f} FPS")
    print("="*50)

if __name__ == "__main__":
    main()