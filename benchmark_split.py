import time
import torch
import os
from navsim.common.dataclasses import TrajectorySampling
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

# ---------------- 配置区域 ----------------
CHECKPOINT_PATH = "/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/exp/recogdrive_stage3_rl_training_16gpus_bs8/lightning_logs/version_0/checkpoints/epoch=9-step=6650.ckpt"
VLM_PATH = "/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/ckpt/ReCogDrive-VLM-8B"
DEVICE_ID = 0
# ------------------------------------------

def main():
    device = torch.device(f"cuda:{DEVICE_ID}")
    print(f"🚀 Starting Split Benchmark on {torch.cuda.get_device_name(device)}...")

    # 1. 初始化 Agent
    print("Loading Model...")
    traj_sampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
    agent = ReCogDriveAgent(
        traj_sampling,
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

    # 获取 InternVL 核心模型
    # agent.backbone 是 RecogDriveBackbone
    # agent.backbone.model 是 HuggingFace InternVLModel
    internvl_model = agent.backbone.model 

    # ---------------- 2. 构造输入 ----------------
    # 模拟输入：1张图, 448x448
    dummy_images = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16).to(device)
    dummy_questions = ["<image>\nPredict trajectory."] 
    dummy_num_patches_list = [1] # 1个patch

    # ---------------- 3. 分段测速 ----------------
    loops = 50
    print(f"\nRunning {loops} loops for breakdown analysis...")

    # --- A. 测试纯视觉编码 (Vision Encoder Only) ---
    # InternVL 的视觉部分叫 vision_model
    start_vis = torch.cuda.Event(enable_timing=True)
    end_vis = torch.cuda.Event(enable_timing=True)
    
    start_vis.record()
    with torch.no_grad():
        for _ in range(loops):
            # 直接调用内部的 vision_model
            # 输入: (B*Num_Patches, C, H, W) -> (1, 3, 448, 448)
            _ = internvl_model.vision_model(dummy_images)
    end_vis.record()
    torch.cuda.synchronize()
    avg_vis = start_vis.elapsed_time(end_vis) / loops

    # --- B. 测试整体 VLM (Total VLM) ---
    # 我们之前测过的那个 478ms
    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)
    
    start_total.record()
    with torch.no_grad():
        for _ in range(loops):
            _ = agent.backbone(dummy_images, dummy_questions, dummy_num_patches_list)
    end_total.record()
    torch.cuda.synchronize()
    avg_total = start_total.elapsed_time(end_total) / loops

    # --- C. 计算 LLM 推理时间 ---
    # LLM 时间 = 总时间 - 视觉时间
    avg_llm = avg_total - avg_vis

    # ---------------- 4. 打印给老师的报告 ----------------
    print("\n" + "="*50)
    print(f"🔬 VLM Internal Breakdown (InternVL-8B)")
    print("="*50)
    print(f"1. Vision Encoder (ViT-6B):  {avg_vis:.2f} ms  ({avg_vis/avg_total*100:.1f}%)")
    print(f"2. LLM Inference (Qwen/Llama): {avg_llm:.2f} ms  ({avg_llm/avg_total*100:.1f}%)")
    print("-" * 50)
    print(f"📦 Total VLM Latency:          {avg_total:.2f} ms")
    print("="*50)
    print("\n[解释]")
    print("Vision Encoder: 负责将图像像素转换为视觉特征 (InternVL 的视觉塔很大，约60亿参数)。")
    print("LLM Inference:  负责处理 Prompt 并结合视觉特征输出 Hidden State。")

if __name__ == "__main__":
    main()