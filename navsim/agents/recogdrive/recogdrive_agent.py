# 'typing' 用于类型注解，能让代码更清晰易读，比如 List, Dict 等
from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer  # 优化器基类
import torch.optim as optim  # 包含了各种优化器，如 AdamW
from torch.optim.lr_scheduler import LRScheduler # 学习率调度器基类
from omegaconf import DictConfig, OmegaConf  # 'omegaconf' 用于管理复杂的配置（比如 .yaml 文件）
from transformers.feature_extraction_utils import BatchFeature  # 来自 Hugging Face 'transformers' 库，用于封装批量数据
import math

# --- 导入 NavSim 和 NuPlan 的相关模块 ---
from navsim.agents.abstract_agent import AbstractAgent  # 智能体（Agent）的抽象基类，继承
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory  # 模拟器使用的数据结构
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder  # 用于构建模型输入（features）和监督目标（targets）的基类
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # NuPlan 库中的轨迹采样配置

# --- 导入 ReCogDrive 项目内部的模块 ---  ReCogDrive 的核心组件
from .utils.internvl_preprocess import load_image  # 加载和预处理 VLM 所需图像的工具函数
from .utils.lr_scheduler import WarmupCosLR  # 自定义的带预热warmup的余弦退火学习率调度器
from .utils.utils import format_number, build_from_configs
from .recogdrive_features import ReCogDriveFeatureBuilder ,TrajectoryTargetBuilder  # 【核心】特征和目标的构建器
from .recogdrive_backbone import RecogDriveBackbone  # 【核心】第一部分：VLM 骨干网络，负责“认知”
from .recogdrive_diffusion_planner import (  # 【核心】第二部分：扩散模型规划器，负责“规划”
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from .recogdrive_rl_algo import GRPOAlgorithm, ReinforcePlusPlusAlgorithm

class ReCogDriveAgent(AbstractAgent):
    """
    ReCogDriveAgent 作为整个模型的核心，格式继承自 AbstractAgent。
    负责：
    1. 初始化 VLM 骨干网络（Backbone）和扩散规划器（Action Head）。
    2. 定义数据处理流程（Feature/Target Builders）。
    3. 串联起 VLM 和规划器，完成从“感知”到“轨迹规划”的完整前向。
    """
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,  # 轨迹采样配置
        vlm_path: Optional[str] = None,  # VLM 预训练模型的路径
        checkpoint_path: Optional[str] = None,  # 整个 Agent（或 Action Head）的预训练权重路径？？
        cam_type: Optional[str] = 'single',
        vlm_type: Optional[str] = 'internvl',  # VLM 的类型（如 'internvl', 'qwen' 等）
        dit_type: Optional[str] = 'small',  # 扩散模型（DiT）的大小（'small' 或 'large'）
        sampling_method: Optional[str] = 'ddim',  # 扩散模型采样方法
        cache_mode: bool = False,  # 是否使用“缓存模式”
        cache_hidden_state: bool = True,  # 【重要】是否缓存 VLM 的隐藏层状态（hidden_state）
        lr: float = 1e-4,  # 学习率
        grpo: bool = False,  # 【重要】是否启用 GRPO
        metric_cache_path: Optional[str] = '',  # GRPO 需要的 PDM 度量缓存路径
        reference_policy_checkpoint: Optional[str] = '',  # GRPO 需要的参考策略（第二阶段训练好的模型）路径
        vlm_size: Optional[str] = 'large',  # VLM 的尺寸标记
        grpo_cfg: Optional[Any] = None,  # 新增这个参数接收命令行传入的配置
        rl_algo_type: str = "grpo", # [新增参数] 算法类型，默认为 'grpo'
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.rl_algo_type = rl_algo_type
        self.grpo_cfg_override = grpo_cfg # 保存一下覆盖配置
        self.backbone = None # VLM 骨干网络实例，默认为 None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        
        # --- 缓存隐藏状态学习（VLM 的加载时机）---
        # ReCogDrive 有两种运行模式，由 cache_hidden_state 控制：
        # 1. 'cache_hidden_state' = True (默认，用于训练规划器):
        #    我们假定 VLM 的输出（hidden_state）已经提前计算好并保存到磁盘了。
        #    在训练时，数据加载器 (ReCogDriveFeatureBuilder) 会直接读取这些缓存的特征。
        #    这样 VLM 就无需在训练规划器时运行，极大节省了显存和时间。
        #    因此，这里的 self.backbone 会是 None。
        
        # 2. 'cache_hidden_state' = False (用于端到端推理 或 缓存特征的生成):
        #    这表示我们需要 VLM 实时运行。
        #    此时，必须在 Agent 内部初始化 VLM 骨干网络 (self.backbone)。
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            device = f"cuda:{local_rank}"
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            # 初始化 VLM 骨干网络
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )
        # --- 初始化扩散规划器 (Action Head) ---
        # 根据 DiT 的大小创不同配置
        # 其中 'input_embedding_dim' 必与 VLM 的输出维度匹配
        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size # 将 VLM 尺寸也存入配置
        # 如果启用 GRPO，则将 GRPO 相关的配置传入
        if self.grpo:
            # 修正后的配置注入逻辑
            if self.grpo_cfg_override is not None:
                for key, value in self.grpo_cfg_override.items():
                    # 1. 获取原始配置中的属性值
                    original_attr = getattr(cfg.grpo_cfg, key, None)
                    # 2. 如果原始属性是一个 Dataclass/对象，且传入的值是字典/DictConfig
                    #    那么我们应该递归更新，而不是直接替换
                    if hasattr(original_attr, "__dataclass_fields__") and (isinstance(value, dict) or isinstance(value, DictConfig)):
                        # 遍历传入的字典，逐个更新属性
                        for sub_key, sub_value in value.items():
                            if hasattr(original_attr, sub_key):
                                setattr(original_attr, sub_key, sub_value)
                            else:
                                print(f"Warning: {sub_key} not found in {key} config, skipping.")
                    else:
                        # 3. 如果是普通类型（float, int），直接替换
                        setattr(cfg.grpo_cfg, key, value)
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda() # 实例化扩散规划器模型，并移动到 GPU
        # 推理时的一些配置
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

        # GRPO 算法初始化逻辑
        self.rl_algo = None
        if self.grpo:
            # 1. 定义算法映射表
            ALGO_MAP = {
                "grpo": GRPOAlgorithm,
                "reinforce_plus_plus": ReinforcePlusPlusAlgorithm,
            }

            # 2. 获取对应的类
            algo_class = ALGO_MAP.get(self.rl_algo_type.lower())
            
            if algo_class is None:
                raise ValueError(f"Unknown rl_algo_type: {self.rl_algo_type}. Supported: {list(ALGO_MAP.keys())}")

            print(f"✅ [Agent] Initializing RL Algorithm: {algo_class.__name__}")

            # 3. 实例化 (接口统一，所以参数一样)
            self.rl_algo = algo_class(cfg.grpo_cfg, self.action_head)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        初始化函数，用于加载预训练权重checkpoint
        这在训练和推理开始时由框架调用
        """
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu",weights_only=False)["state_dict"]
            model_dict = self.state_dict() #获取当前模型（Agent）的状态字典
            filtered_ckpt = {}
            # --- 权重过滤逻辑 ---
            # 经常地，保存的 .ckpt 里的 key 会带有前缀（比如 'agent.' 或 'model.'）
            # 而 self.state_dict() 的 key 没有前缀。
            # 这里的代码是为了剥离前缀，使 .ckpt 中的 key 与当前模型的 key 匹配
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            # 'strict=False' ----允许只加载部分权重
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        【重要】告诉训练框架如何构建“输入特征”。
        返回一个 'ReCogDriveFeatureBuilder' 实例。这个 Builder 的行为取决于 `__init__` 中的配置：
        - 如果 'cache_hidden_state' = True:
          它会去加载预先计算好的 VLM 隐藏状态 (last_hidden_state) 和其他状态特征。
        - 如果 'cache_hidden_state' = False:
          它会加载图像的 *路径* (image_path_tensor) 和其他状态特征。
          真正的图像加载和 VLM 推理将在 `forward` 方法中进行。
        """    
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            cache_mode=self.cache_mode,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        
        """
        【模型的核心前向传播逻辑】
        输入:
        - features (Dict): 来自 FeatureBuilder 的输入数据。
        - targets (Dict): 来自 TargetBuilder 的目标数据（仅训练时）。
        - tokens_list: (用于 GRPO) PDM 度量相关的 token
        输出:
        - Dict: 包含损失（训练时）或预测轨迹（推理时）的字典。
        """
        # --- 数据准备：将所有张量移动到 GPU ---
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype # 获取规划器的数据类型（如 float16）

        history_trajectory = features["history_trajectory"].cuda() # 历史轨迹，举例：形状: (B, 4, 3)，4 表示时间步数
        high_command_one_hot = features["high_command_one_hot"].cuda() # 高层导航指令 (e.g., [0,1,0] for 'go straight')
        
        # 确保有 batch 维度（推理时可能没有
        if history_trajectory.ndim == 2: history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1: high_command_one_hot = high_command_one_hot.unsqueeze(0)
        # --- 获取 VLM 的隐藏状态 ---
        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            # 【No-Cache 模式】实时运行 VLM
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            # 2a. 从 'features' 中获取图像路径张量
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
            # 2b. 解码图像路径张量为字符串列表
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            # 2c. 加载和预处理图像
            pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
            
            # 2d. 构建文本提示（prompts）
            # 1 指令映射
            navigation_commands = ['turn left', 'go straight', 'turn right'] 
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]
            # 2 存储每个 batch 样本的 prompt
            questions = [] 
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]
                # 格式化历史轨迹，使其成为文本
                history_str = ' '.join([
                    f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                    f'{format_number(history_trajectory_sample[j, 1].item())}, '
                    f'{format_number(history_trajectory_sample[j, 2].item())})'
                    for j in range(history_trajectory_sample.shape[0])
                ])
                
                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")
            # 2e. 运行 VLM (Backbone)
            # VLM 接收图像 (pixel_values_cat) 和 文本 (questions)，输出认知特征
            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        # --- 3. 准备规划器的“状态”输入 ---
        status_feature = features["status_feature"].cuda()  # 车辆状态 (速度、加速度等)
        # 确保有 batch 维度
        if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2: last_hidden_state = last_hidden_state.unsqueeze(0)

        # 历史轨迹展平 (B, 4, 3) -> (B, 12)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        # 拼接车辆状态和历史轨迹，作为规划器的“低维状态”输入
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        
        
        # --- 4. 运行扩散规划器 (Action Head) ---
        # 根据是训练、GRPO训练还是推理，调用 'action_head' 不同方法
        # if self.training and not self.grpo:
        #     # 【阶段二：标准训练模式】
        #     # 将 'input_state' 和 VLM 特征（作为 condition）
        #     # 以及目标轨迹 'targets["trajectory"]' 传给规划器
        #     action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
        #     # action_head 的 forward 会计算 diffusion loss
        #     return self.action_head(last_hidden_state, action_inputs)
        
        # elif self.training and self.grpo:
        #     # 【阶段三：GRPO 训练模式】
        #     action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype)})
        #     # 调用 GRPO 特定的前向传播
        #     return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        
        # else: 
        #     # 【推理模式】
        #     # 只提供状态和 VLM 特征，不提供目标 'action'
        #     action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
        #     # 调用 'get_action' 来采样生成轨迹
        #     return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)
        
        # 构造 input batch
        # 这里我统一了一下，把 SFT 和 RL 的输入构造稍微整理得整洁一点
        if self.training:
            action_inputs = BatchFeature(data={
                "state": input_state.to(model_dtype), 
                "his_traj": history_trajectory_reshaped.to(model_dtype), 
                "status_feature": status_feature.to(model_dtype), 
                "action": targets["trajectory"].to(model_dtype)
            })

            if not self.grpo:
                # SFT 路径
                return self.action_head(last_hidden_state, action_inputs)
            else:
                # GRPO 路径
                # 不再调用 self.action_head.forward_grpo
                # 而是调用 self.rl_algo.compute_loss
                return self.rl_algo.compute_loss(
                    actor_model=self.action_head,
                    vl_features=last_hidden_state,
                    action_input=action_inputs,
                    tokens_list=tokens_list
                )
        else:
            # 推理路径不变
                action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype)})
                return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)
    def compute_trajectory(self, features: Dict[str, torch.Tensor]) -> Trajectory:
        """
        在模拟器中进行推理（evaluation）的入口函数。
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0) # 获取预测轨迹
        return Trajectory(poses) # 封装成 Trajectory 对象

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        """
        一个可视化的推理函数，其中它手动执行了 'get_feature_builders' 的过程。
        """
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        
        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        由训练框架（Lightning）调用的损失计算函数。
        'predictions' 是 'forward' 方法的返回值
        """
        if self.training and self.grpo:
            # GRPO 模式下，'forward_grpo' 返回的 'predictions' 已经是一个损失字典
            return predictions
        elif self.training:
            # 标准训练模式下，'forward' 返回的 'predictions' 是一个包含 '.loss' 属性的对象
            return predictions.loss
        else:
            # 评估模式下，计算一个简单的 L1 损失作为度量（metric）
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        """
        同上由训练框架（Lightning）调用，用于设置优化器和学习率调度器。
        """
        # --- 优化器 ---
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        optimizer = build_from_configs(optim, optimizer_cfg, params=self.action_head.parameters())
        # --- 学习率调度器 Scheduler ---
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths
# --- 辅助函数：创建扩散规划器的配置 ---
def make_recogdrive_config(
    size: str,  # 预设的尺寸，如 "small", "large"
    *,
    action_dim: int,  # 动作维度（x, y, heading），所以是 3
    action_horizon: int,  # 动作视界（预测未来多少步），所以是 8
    input_embedding_dim: int,  # 输入 VLM 嵌入的维度（384 或 1536）
    sampling_method: str = 'ddim',  # 采样方法
    num_inference_steps: int = 5,  # 推理步数
    grpo: bool = False,  # 是否启用 GRPO
    model_dtype: str = "float16",  # 模型数据类型
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.
    
    一个工厂函数 (Factory Function)，用于创建 ReCogDriveDiffusionPlannerConfig 对象。
    这使得创建配置更加简单，只需要指定 'size' 和一些关键参数。
    
    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
