from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone
from .utils.internvl_preprocess import load_image

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False, ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
        """
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras
        '''流水线 A: 数值数据处理 , 把物理世界的数字转换成 Tensor '''
        # 提取历史轨迹 (last 4 timesteps)
        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        # 提取高层命令
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        # 拼接状态特征 (指令 + 速度 + 加速度) -> 供后续 Diffusion Planner 使用
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            
            path_as_ordinals = [ord(char) for char in image_path]
            
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            
            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
        else:
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            '''流水线 B: 图像数据处理 , 把硬盘上的 JPG 图片变成模型能吃的 Tensor...这部分代码是为了配合 VLM 的输入格式要求（它支持一次输入多张不同的图片，且每张图片的切片数可能不一样）
            核心操作：
                 oad_image:
                        1.dynamic_preprocess 负责计算切几块，并返回切好的一堆 PIL.Image 对象(use_thumbnail=True 表示一定要包含那张全局缩略图)
                        2.对每一张小图做归一化 (Normalize) 和 转 Tensor
                        3.把所有切块拼接成一个大 Tensor,形状为 (N, 3, 448, 448),N 是切块数量切片数+1
            '''
            pixel_values = load_image(str(cameras[-1].cam_f0.image)).unsqueeze(0) #增加 batch 维度，变成 (1, N, 3, 448, 448)

            pixel_values_squeezed = pixel_values.squeeze(1) #squeeze(dim) 只有在该维度大小等于 1 时才生效
            num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
            pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)
            '''流水线 C: 提示词工程 , 把数字（One-Hot 编码、坐标数值）翻译成 VLM 能听懂的自然语言
            prompt 样子（示例）：
                <image> As an autonomous driving system, predict the vehicle's trajectory based on:
                1.Visual perception from front camera view
                2.Historical motion context (last 4 timesteps): - t-3: (0.0, 0.0, 0.0) ...
                3.Active navigation command: [GO STRAIGHT]
            Output requirements:
                - Predict 8 future trajectory points
                - Each point format: (x:float, y:float, heading:float)
            '''

            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
            history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
            
            prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
            output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
            questions = [f"{prompt}{output_requirements}"]

            outputs = self.backbone(pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                "status_feature": status_feature.cpu(),
            }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
