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
        
        # 1. 注入采样参数 (对应原代码 self.denoised_clip_value = ... 等)
        sampling_params = [
            "denoised_clip_value", "eval_randn_clip_value", "randn_clip_value",
            "final_action_clip_value", "eps_clip_value", "eval_min_sampling_denoising_std",
            "min_sampling_denoising_std", "min_logprob_denoising_std"
        ]
        print("[GRPO] Injecting sampling hyperparameters into Actor Model...")
        for param_key in sampling_params:
            if hasattr(config, param_key):
                setattr(model_template, param_key, getattr(config, param_key))

        # 2. 初始化缓存和评分器 (对应原代码 self.metric_cache_loader = ... 等)
        self.clip_advantage_lower_quantile = config.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = config.clip_advantage_upper_quantile
        self.gamma_denoising = config.gamma_denoising
        
        print(f"[GRPO] Loading Metric Cache from {config.metric_cache_path}...")
        self.metric_cache_loader = MetricCacheLoader(Path(config.metric_cache_path))
        
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(proposal_sampling, config.scorer_config)
        
        # 3. 加载权重 (对应原代码的 try...torch.load... 块)
        if config.reference_policy_checkpoint:
            print(f"[GRPO] Loading checkpoint from {config.reference_policy_checkpoint}")
            try:
                state_dict = torch.load(config.reference_policy_checkpoint, map_location="cpu")["state_dict"]
                model_dict = model_template.state_dict()
                filtered_ckpt = {}
                for k, v in state_dict.items():
                    k2 = k[len("agent.action_head."):] if k.startswith("agent.action_head.") else k
                    if k2 in model_dict and v.shape == model_dict[k2].shape:
                        filtered_ckpt[k2] = v
                    else:
                        pass # 忽略不匹配的键
                
                # 加载到当前的主模型 (Student)
                model_template.load_state_dict(filtered_ckpt, strict=True)
                print("[GRPO] Successfully loaded weights into Actor Model.")
            except FileNotFoundError:
                print(f"[GRPO] Warning: Checkpoint not found at {config.reference_policy_checkpoint}")
            except Exception as e:
                print(f"[GRPO] Error loading checkpoint: {e}")

        # 4. 创建 Reference Model (对应原代码 self.old_policy = copy.deepcopy(self))
        print("[GRPO] Initializing Reference Model (Old Policy)...")
        self.ref_model = copy.deepcopy(model_template)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        print("[GRPO] Initialization Complete.")

    def compute_loss(self, actor_model: ReCogDriveDiffusionPlanner, vl_features: torch.Tensor, 
                     action_input: BatchFeature, tokens_list: List[str], sample_time: int = 8, 
                     bc_coeff: float = 0.1, use_bc_loss: bool = True) -> BatchFeature:
        # 确保 Reference Model 在正确的设备上
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
        
        # 4. 优势计算 (GRPO核心)
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
            except:
                rewards.append(0.0)
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()