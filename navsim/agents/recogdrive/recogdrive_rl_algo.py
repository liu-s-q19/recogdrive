import copy
import lzma
import pickle
import torch
import torch.nn as nn
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from transformers.feature_extraction_utils import BatchFeature

from .recogdrive_diffusion_planner import GRPOConfig, ReCogDriveDiffusionPlanner

class RLAlgorithm:
    def compute_loss(self, actor_model: nn.Module, inputs: BatchFeature, tokens_list: List[str]) -> BatchFeature:
        raise NotImplementedError

class GRPOAlgorithm(RLAlgorithm):
    def __init__(self, config: GRPOConfig, model_template: ReCogDriveDiffusionPlanner):
        super().__init__()
        self.cfg = config
        
        # 注入采样参数
        sampling_params = [
            "denoised_clip_value", "eval_randn_clip_value", "randn_clip_value",
            "final_action_clip_value", "eps_clip_value", "eval_min_sampling_denoising_std",
            "min_sampling_denoising_std", "min_logprob_denoising_std"
        ]
        print("Injecting sampling hyperparameters into Actor Model...")
        for param_key in sampling_params:
            if hasattr(config, param_key):
                setattr(model_template, param_key, getattr(config, param_key))

        # 初始化缓存和评分器
        self.clip_advantage_lower_quantile = config.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = config.clip_advantage_upper_quantile
        self.gamma_denoising = config.gamma_denoising

        print(f"Loading Metric Cache from {config.metric_cache_path}")
        self.metric_cache_loader = MetricCacheLoader(Path(config.metric_cache_path))
        
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, config.scorer_config)
        
        # 3. 加载权重
        if config.reference_policy_checkpoint:
            print(f"[GRPO] Loading checkpoint from {config.reference_policy_checkpoint}")
            try:
                state_dict = torch.load(config.reference_policy_checkpoint, map_location="cpu",weights_only=False)["state_dict"]
                model_dict = model_template.state_dict()
                filtered_ckpt = {}
                for k, v in state_dict.items():
                    k2 = k[len("agent.action_head."):] if k.startswith("agent.action_head.") else k
                    if k2 in model_dict and v.shape == model_dict[k2].shape:
                        filtered_ckpt[k2] = v
                    else:
                        pass # 忽略不匹配键
                # 加载到当前的主模型
                model_template.load_state_dict(filtered_ckpt, strict=True)
                print("Successfully loaded weights into Actor Model.")
            except FileNotFoundError:
                print(f"Warning: Checkpoint not found at {config.reference_policy_checkpoint}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        # 4. 创建 Reference Model
        print("Initializing Reference Model (Old Policy)...")
        self.ref_model = copy.deepcopy(model_template)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        print("Initialization Complete.")

    def compute_loss(self, actor_model: ReCogDriveDiffusionPlanner, vl_features: torch.Tensor, 
                     action_input: BatchFeature, tokens_list: List[str], sample_time: int = 8, 
                     bc_coeff: float = 0.1, use_bc_loss: bool = True) -> BatchFeature:
        # Reference Model
        if self.ref_model.device != actor_model.device:
            self.ref_model.to(actor_model.device)

        B = vl_features.shape[0]
        G = sample_time
        
        # 1. 复制数据
        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        
        # 2. 采样
        chains, trajs = actor_model.sample_chain(
            vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False
        )

        # 3. 奖励计算
        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        rewards = self._reward_fn(trajs, tokens_rep)
        
        # 4. 优势计算
        rewards_matrix = rewards.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)
        
        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
        
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
        adv_weighted_flat = (adv_steps * discount).reshape(-1)

        # 5. Policy Loss
        log_probs = actor_model.get_logprobs(
            vl_features_rep, his_traj_rep, status_feature_rep, chains, deterministic=False
        )
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        # 6. BC Loss
        bc_loss = torch.tensor(0.0, device=actor_model.device)
        if use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.ref_model.sample_chain(
                    vl_features, action_input.his_traj, action_input.status_feature, deterministic=False
                )
            bc_logp = actor_model.get_logprobs(
                vl_features, action_input.his_traj, action_input.status_feature, teacher_chains, deterministic=False
            )
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + bc_coeff * bc_loss

        return BatchFeature(data={"loss": total_loss, "reward": rewards.mean(), "policy_loss": policy_loss, "bc_loss": bc_loss})

    def _reward_fn(self, pred_traj: torch.Tensor, tokens_list: List[str]) -> torch.Tensor:
        unique_tokens = set(tokens_list)
        cache_dict = {}
        for token in unique_tokens:
            if token in self.metric_cache_loader.metric_cache_paths:
                path = self.metric_cache_loader.metric_cache_paths[token]
                with lzma.open(path, 'rb') as f:
                    cache_dict[token] = pickle.load(f)
            else:
                 return torch.zeros(len(tokens_list), device=pred_traj.device) # Fallback

        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
        for i, token in enumerate(tokens_list):
            if token not in cache_dict:
                rewards.append(0.0)
                continue
            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[token]
            try:
                pdm_result = pdm_score(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    future_sampling=self.simulator.proposal_sampling,
                    simulator=self.simulator,
                    scorer=self.train_scorer,
                )
                rewards.append(asdict(pdm_result)["score"])
            except Exception as e:
                # ！！！在这里加个 print，看看究竟报了什么错！！！
                print(f"[ERROR in Reward]: {e}")
                # print(f"Scorer Config Type: {type(self.train_scorer.config)}") 
                rewards.append(0.0)
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()
    



class ReinforcePlusPlusAlgorithm(GRPOAlgorithm):
    """
    Reinforce++ 算法实现 (https://arxiv.org/abs/2501.03262)。
    
    继承关系:
    RLAlgorithm -> GRPOAlgorithm -> ReinforcePlusPlusAlgorithm
    
    该类直接复用 GRPOAlgorithm 的:
    - __init__ (模型加载、参数注入)
    - _reward_fn (PDM 评分逻辑)
    - metric_cache_loader, simulator 等基础设施
    
    仅重写:
    - compute_loss: 修改了 Advantage 的计算方式 (Group Center + Batch Norm)
    """

    def compute_loss(
        self, 
        actor_model: ReCogDriveDiffusionPlanner, 
        vl_features: torch.Tensor, 
        action_input: BatchFeature, 
        tokens_list: List[str], 
        sample_time: int = 8, 
        bc_coeff: float = 0.1, 
        use_bc_loss: bool = True
    ) -> BatchFeature:
        
        # ---------------------------------------------------------------------
        # 0. 基础准备
        # ---------------------------------------------------------------------
        # 确保 Reference Model 设备同步 (这是常见的 bug 源，检查一下很必要)
        if self.ref_model.device != actor_model.device:
            self.ref_model.to(actor_model.device)

        B = vl_features.shape[0]
        G = sample_time # Group Size (每个 Prompt 采样的数量)
        
        # ---------------------------------------------------------------------
        # 1. 数据复制与采样 (Sampling)
        # ---------------------------------------------------------------------
        # 将 Batch 扩展 G 倍，构造 (B*G) 的输入
        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        
        # 前向采样：生成 Chains (用于算概率) 和 Trajs (用于算分)
        chains, trajs = actor_model.sample_chain(
            vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False
        )

        # ---------------------------------------------------------------------
        # 2. 奖励计算 (Reward Calculation)
        # ---------------------------------------------------------------------
        # 构造对应的 Token 列表并计算奖励
        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        rewards = self._reward_fn(trajs, tokens_rep) # 返回 shape: (B*G, )
        
        # ---------------------------------------------------------------------
        # 3. 优势计算 (Advantage Calculation) - Reinforce++ 核心逻辑
        # ---------------------------------------------------------------------
        # 变形为 (Batch, Group)
        rewards_matrix = rewards.view(B, G)
        
        # [Step 3.1] 组内去中心化 (Group De-centering / Baseline Subtraction)
        # 计算每个 Prompt 组内的均值。
        # 逻辑：Adv_temp = R_i - Mean(R_group)
        # 作用：消除不同 Prompt 难度差异带来的方差。
        group_mean = rewards_matrix.mean(dim=1, keepdim=True)
        advantages = rewards_matrix - group_mean 
        
        # [Step 3.2] 批次标准化 (Batch Normalization)
        # 展平回 (B*G, )
        advantages = advantages.view(-1)
        
        # 逻辑：Adv_final = (Adv_temp - BatchMean) / BatchStd
        # 作用：保证梯度幅值稳定，类似于 BN 层的作用。
        # 注意：使用 detach() 防止梯度回传到统计量。
        batch_mean = advantages.mean()
        batch_std = advantages.std() + 1e-8 # 加 epsilon 防止除零
        
        advantages = (advantages - batch_mean) / batch_std
        advantages = advantages.detach() 
        
        # [Step 3.3] 优势裁剪 (Advantage Clipping)
        # 逻辑：截断过大的优势值，防止单个样本主导梯度。
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)
        
        # [Step 3.4] 去噪步衰减 (Denoising Step Discounting)
        # 逻辑：对扩散模型不同时间步施加权重 (越接近成品权重越大)
        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
        
        # 广播 Advantage 到每个时间步
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
        adv_weighted_flat = (adv_steps * discount).reshape(-1)

        # ---------------------------------------------------------------------
        # 4. 损失计算 (Policy Loss)
        # ---------------------------------------------------------------------
        # 计算整条链的 Log Probability
        log_probs = actor_model.get_logprobs(
            vl_features_rep, his_traj_rep, status_feature_rep, chains, deterministic=False
        )
        
        # 空间维度平均 (Mean over H, W)，防止数值过大
        log_probs = log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        
        # 标准 Policy Gradient Loss: - Mean( log_prob * advantage )
        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        # ---------------------------------------------------------------------
        # 5. 辅助损失 (BC Loss)
        # ---------------------------------------------------------------------
        bc_loss_val = torch.tensor(0.0, device=actor_model.device)
        if use_bc_loss:
            with torch.no_grad():
                teacher_chains, _ = self.ref_model.sample_chain(
                    vl_features, action_input.his_traj, action_input.status_feature, deterministic=False
                )
            bc_logp = actor_model.get_logprobs(
                vl_features, action_input.his_traj, action_input.status_feature, teacher_chains, deterministic=False
            )
            # BC Logp 处理
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss_val = -bc_logp.mean()
            
            total_loss = total_loss + bc_coeff * bc_loss_val

        return BatchFeature(data={
            "loss": total_loss, 
            "reward": rewards.mean(), 
            "policy_loss": policy_loss, 
            "bc_loss": bc_loss_val
        })