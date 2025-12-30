import os
import sys
import json
import torch
import hydra
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import math
import re
import numpy as np
import traceback
from PIL import Image

# --- 0. 环境变量自动注入 ---
def setup_environment():
    data_root = os.getenv("NAVSIM_DATA_ROOT")
    if not data_root:
        default_path = "/nfs/dataset-ofs-prediction/rl_lab/leidianqiao/code/recogdrive/data/navsim"
        if os.path.exists(default_path):
            print(f"[Auto-Setup] NAVSIM_DATA_ROOT not set. Using default: {default_path}")
            os.environ["NAVSIM_DATA_ROOT"] = default_path
            data_root = default_path
        else:
            print("[Error] Please export NAVSIM_DATA_ROOT!")
            return False

    if "OPENSCENE_DATA_ROOT" not in os.environ:
        print(f"[Auto-Setup] OPENSCENE_DATA_ROOT not set. Syncing with NAVSIM_DATA_ROOT.")
        os.environ["OPENSCENE_DATA_ROOT"] = data_root

    if "NUPLAN_MAPS_ROOT" not in os.environ:
        maps_path = Path(data_root) / "maps"
        if maps_path.exists():
            print(f"[Auto-Setup] NUPLAN_MAPS_ROOT not set. Auto-setting to: {maps_path}")
            os.environ["NUPLAN_MAPS_ROOT"] = str(maps_path)
        else:
            print(f"[Warning] Maps folder not found at {maps_path}. Map loading might fail!")
    
    return True

if not setup_environment():
    sys.exit(1)

# --- 1. 导入 ---
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone
# 导入底层处理函数
from navsim.agents.recogdrive.utils.internvl_preprocess import dynamic_preprocess, build_transform
from navsim.agents.recogdrive.utils.utils import format_number

# --- 2. 自定义高效图片加载函数 ---
def process_image_from_array(image_array, input_size=448, max_num=12):
    image = Image.fromarray(image_array).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# --- 3. System Message (简化版) ---
REASONING_SYSTEM_MESSAGE = """
You are an autonomous driving assistant. 
Your task is to analyze the scene and explain the expert driver's action concisely.
"""

# --- 4. 核心逻辑函数 ---
def get_future_behavior(current_status, future_trajectory):
    if future_trajectory is None: return "Unknown Action"
    if hasattr(future_trajectory, 'poses'): poses = future_trajectory.poses
    else: poses = future_trajectory
    if len(poses) < 5: return "Unknown Action"

    future_idx = min(len(poses) - 1, 29) 
    local_pose = poses[future_idx] 
    dx, dy, d_theta = local_pose[0], local_pose[1], local_pose[2]
    
    dist = math.sqrt(dx**2 + dy**2)
    dt = (future_idx + 1) * 0.1 
    avg_vel = dist / (dt + 1e-6)
    curr_vel = math.sqrt(current_status.ego_velocity[0]**2 + current_status.ego_velocity[1]**2)
    behavior = []

    if avg_vel > curr_vel + 1.5: behavior.append("Accelerate")
    elif avg_vel < curr_vel - 1.5: behavior.append("Decelerate")
    elif curr_vel < 0.5 and avg_vel < 0.5: behavior.append("Remain Stationary")
    else: behavior.append("Maintain Speed")

    if d_theta > 0.15: behavior.append("Turn Left")
    elif d_theta < -0.15: behavior.append("Turn Right")
    else:
        if dy > 1.5: behavior.append("Lane Change Left")
        elif dy < -1.5: behavior.append("Lane Change Right")
        else: behavior.append("Keep Lane")

    return " and ".join(behavior)

def try_repair_response(text):
    """
    更强大的清洗逻辑：去除嵌套标签，提取核心内容
    """
    # 1. 基础清洗：去除幻觉标签和坐标
    text = text.replace("</box>", "").replace("<ref>", "").replace("</ref>", "")
    text = text.replace("</p>", "").replace("<a>", "").replace("</a>", "")
    text = re.sub(r'\[.*?\]', '', text) # 去除所有方括号内容 [0, 0.1...]
    
    # 2. 提取 Risk (优先找标签，找不到找关键词)
    risk = "Unknown"
    risk_match = re.search(r'<risk_level>(.*?)</risk_level>', text, re.DOTALL | re.IGNORECASE)
    if risk_match:
        risk = risk_match.group(1).strip()
    else:
        # 关键词兜底
        lower_text = text.lower()
        if "high risk" in lower_text: risk = "High"
        elif "medium risk" in lower_text: risk = "Medium"
        elif "low risk" in lower_text: risk = "Low"
    
    # 3. 提取 Perception (解决嵌套问题)
    # 策略：如果找不到 <perception>，尝试找 "Perception:" 文本，或者从 Reasoning 里拆
    perp = "Implied in reasoning"
    perp_match = re.search(r'<perception>(.*?)</perception>', text, re.DOTALL | re.IGNORECASE)
    
    if perp_match:
        perp = perp_match.group(1).strip()
    else:
        # 尝试查找嵌套在 reasoning 里的 perception
        # 有时候模型写成: <reasoning> <perception> xxx </perception> ...
        pass # 正则已经覆盖了这种情况，如果还提取不到，说明真的没写

    # 4. 提取 Reasoning (去除嵌套在里面的其他标签)
    reason = text
    reason_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
        # 【关键】：如果 reasoning 里面包了 <perception>，把它剔除掉
        reason = re.sub(r'<risk_level>.*?</risk_level>', '', reason, flags=re.DOTALL)
        reason = re.sub(r'<perception>.*?</perception>', '', reason, flags=re.DOTALL)
        reason = reason.strip()
    
    # 5. 最终清洗：去除多余换行
    perp = " ".join(perp.split())
    reason = " ".join(reason.split())

    return f"<risk_level>{risk}</risk_level>\n<perception>{perp}</perception>\n<reasoning>{reason}</reasoning>", True

# --- 主入口 ---
@hydra.main(config_path="../../navsim/planning/script/config/common/train_test_split", config_name="navtrain") 
def main(cfg: DictConfig):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device_str = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    
    model_path = cfg.get("model_path", "/path/to/your/InternVL-weights")
    if rank == 0:
        print(f"========================================")
        print(f"Process Rank: {rank}/{world_size}")
        print(f">> Model Path: {model_path}")
    
    output_file = f"reasoning_gt_part_{rank}.json" 
    skipped_file = f"skipped_gt_part_{rank}.json"
    data_root = os.getenv("NAVSIM_DATA_ROOT")
    
    # 1. 模型加载
    try:
        backbone = RecogDriveBackbone(
            model_type='internvl', checkpoint_path=model_path, device=device_str
        )
        backbone.eval()
        # 清空 System Message，防止干扰
        if hasattr(backbone.model, 'system_message'):
            backbone.model.system_message = REASONING_SYSTEM_MESSAGE
    except Exception as e:
        print(f"[Rank {rank}] Model Load Error: {e}")
        return

    # 2. 路径对齐
    navsim_root = Path(data_root)
    log_search_path = navsim_root / "navsim_logs" / "trainval"
    if not log_search_path.exists(): log_search_path = navsim_root / "navsim_logs"
    
    sensor_blobs_path = navsim_root / "sensor_blobs" / "trainval"
    if not sensor_blobs_path.exists(): sensor_blobs_path = navsim_root / "sensor_blobs"

    all_local_logs = list(log_search_path.glob("*.pkl"))
    if len(all_local_logs) == 0:
        print(f"[ERROR] No .pkl files found in {log_search_path}!")
        return

    if rank == 0: 
        print(f"DEBUG: Logs Path: {log_search_path}")
        print(f"DEBUG: Blobs Path: {sensor_blobs_path}")

    all_tokens_str = [f.stem for f in all_local_logs]
    
    # 3. 过滤器设置
    actual_filter = cfg.scene_filter
    if world_size == 1:
        if rank == 0: print(f"DEBUG: Local Mode -> Loading all {len(all_tokens_str)} logs.")
        actual_filter = OmegaConf.create({
            "log_names": all_tokens_str,
            "tokens": None,             
            "scene_blacklist": None,    
            "max_scenes": None,           
            "num_frames": 200,            
            "frame_interval": 200,        
            "start_frame_index": 0,
            "timestamp_threshold_s": None,
            "has_route": False,           
            "num_history_frames": 4,      
            "num_future_frames": 10,
            "min_future_frames": None,
            "camera_type": None,
            "lidar_type": None
        })

    sensor_config = SensorConfig.build_all_sensors()

    # 4. Loader
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=log_search_path,
        scene_filter=actual_filter, 
        sensor_config=sensor_config, 
    )
    
    all_tokens = scene_loader.tokens
    my_tokens = all_tokens[rank::world_size]
    
    if world_size == 1:
        my_tokens = my_tokens[:5] # 【正式跑全量时，请注释掉这行】
        if rank == 0: print(f"DEBUG: Processing {len(my_tokens)} tokens")

    # 5. Loop
    reasoning_database = {}
    tokens_to_process = my_tokens
    pbar = tqdm(tokens_to_process, desc=f"R{rank}", disable=(rank!=0), position=rank)
    
    valid_count = 0

    for token in pbar:
        try:
            scene = scene_loader.get_scene_from_token(token)
            
            future_trajectory = None
            try: future_trajectory = scene.get_future_trajectory(num_trajectory_frames=30)
            except: pass

            agent_input = scene.get_agent_input()
            gt_action_desc = get_future_behavior(agent_input.ego_statuses[-1], future_trajectory)
            
            if gt_action_desc == "Unknown Action": continue

            # --- 内存读取 ---
            try:
                cam_data = agent_input.cameras[-1].cam_f0
                image_array = cam_data.image
                
                if True:
                    print(f"\n📸 Capturing target image for token: {token}")
                    debug_save_path = f"{token}.jpg"
                    Image.fromarray(image_array).save(debug_save_path)
                    print(f"✅ Image saved to: {os.path.abspath(debug_save_path)}")
                    print(f"👉 You can open it in VS Code by running: code {debug_save_path}\n")

                if image_array is None: raise ValueError("Image None")
                pixel_values = process_image_from_array(image_array).to(torch.bfloat16).cuda()
            except Exception as e_img:
                if world_size == 1: print(f"[Skip] Image Error: {e_img}")
                continue

            cmd_idx = 1
            high_command = agent_input.ego_statuses[-1].driving_command
            for i, val in enumerate(high_command):
                if val == 1: cmd_idx = i; break
            command_str = ['TURN LEFT', 'GO STRAIGHT', 'TURN RIGHT'][cmd_idx]

            hist_traj = torch.tensor([[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in agent_input.ego_statuses[:4]])
            hist_str = " ".join([f't-{3-i}:({format_number(hist_traj[i,0].item())},{format_number(hist_traj[i,1].item())})' for i in range(4)])

            # ================= 核心修改：ONE-SHOT PROMPT =================
            # 给出范例，强制简洁，强制结构
            prompt = (
                f"<image>\n"
                f"You are an AI analyzing human driving.\n"
                f"COMMAND: {command_str}\n"
                f"ACTION: {gt_action_desc}\n\n"
                f"Respond in this EXACT XML format (Concise, <30 words per section):\n"
                f"<risk_level>Low/Medium/High</risk_level>\n"
                f"<perception>Key objects (Traffic light state, Front car status, Obstacles).</perception>\n"
                f"<reasoning>Directly explain why the action was taken based on perception.</reasoning>\n\n"
                f"Example:\n"
                f"<risk_level>Medium</risk_level>\n"
                f"<perception>Red traffic light ahead. Lead vehicle is braking.</perception>\n"
                f"<reasoning>The driver decelerated to stop safely behind the lead vehicle at the red light.</reasoning>\n\n"
                f"Your Output:"
            )

            # 限制 max_new_tokens 只有 200，逼迫模型写短句
            generation_config = dict(
                num_beams=1, 
                max_new_tokens=256, 
                do_sample=False, 
                repetition_penalty=1.1
            )
            
            response = backbone.model.chat(
                tokenizer=backbone.tokenizer, pixel_values=pixel_values,
                question=prompt, generation_config=generation_config
            )
            
            final_response, _ = try_repair_response(response)
            
            if len(final_response) > 20:
                reasoning_database[token] = final_response
                valid_count += 1
                if valid_count % 10 == 0:
                    with open(output_file, 'w') as f: json.dump(reasoning_database, f, indent=4)

        except Exception as e:
            if world_size == 1: 
                print(f"\n[ERROR] Crash on token {token}:")
                traceback.print_exc()
            continue

    with open(output_file, 'w') as f: json.dump(reasoning_database, f, indent=4)
    if rank == 0: print(f"[Rank 0] Finished. Valid samples: {len(reasoning_database)}")

if __name__ == "__main__":
    main()