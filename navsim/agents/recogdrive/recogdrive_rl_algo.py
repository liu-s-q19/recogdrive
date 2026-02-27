import copy
import lzma
import pickle
import torch
import torch.distributed as dist
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
    manual_optimization: bool = False

    def compute_loss(self, actor_model: nn.Module, inputs: BatchFeature, tokens_list: List[str]) -> BatchFeature:
        raise NotImplementedError

class ReinforceAlgorithm(RLAlgorithm):
    def __init__(self, config: GRPOConfig, model_template: ReCogDriveDiffusionPlanner):
        super().__init__()
        self.cfg = config
        self.reference_policy_loaded = False
        
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
        cache_root = Path(config.metric_cache_path)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            cache_paths_obj = None
            if rank == 0:
                cache_paths_obj = MetricCacheLoader(cache_root).metric_cache_paths
            obj_list = [cache_paths_obj]
            dist.broadcast_object_list(obj_list, src=0)

            self.metric_cache_loader = MetricCacheLoader.__new__(MetricCacheLoader)
            self.metric_cache_loader._file_name = "metric_cache.pkl"
            self.metric_cache_loader.metric_cache_paths = obj_list[0]
        else:
            self.metric_cache_loader = MetricCacheLoader(cache_root)
        
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
                self.reference_policy_loaded = True
            except FileNotFoundError:
                print(f"Warning: Checkpoint not found at {config.reference_policy_checkpoint}")
                self.reference_policy_loaded = False
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                self.reference_policy_loaded = False

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
        missing_tokens = []
        for token in unique_tokens:
            if token in self.metric_cache_loader.metric_cache_paths:
                cache_dict[token] = self.metric_cache_loader.get_from_token(token)
            else:
                missing_tokens.append(token)

        if missing_tokens:
            missing_preview = missing_tokens[:5]
            print(
                f"[WARN][Reward] Missing metric cache for {len(missing_tokens)} tokens. "
                f"Examples: {missing_preview}"
            )

        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
        error_count = 0
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
                error_count += 1

        self._last_reward_debug = {
            "num_samples": len(tokens_list),
            "num_unique_tokens": len(unique_tokens),
            "num_missing_unique_tokens": len(missing_tokens),
            "missing_unique_token_frac": (len(missing_tokens) / max(1, len(unique_tokens))),
            "reward_error_count": error_count,
        }
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()


class GRPOClipAlgorithm(ReinforceAlgorithm):
    """
    GRPO/PPO-clip style algorithm with:
    - single rollout collection per training step
    - multi-epoch minibatch optimization on fixed rollout
    """

    manual_optimization: bool = True

    def __init__(self, config: GRPOConfig, model_template: ReCogDriveDiffusionPlanner):
        super().__init__(config, model_template)
        self.clip_epsilon = config.clip_epsilon
        self.ppo_epochs = config.ppo_epochs
        self.mini_batch_size = config.mini_batch_size
        self.max_grad_norm = config.max_grad_norm
        self.target_kl = config.target_kl
        self.sample_time = config.sample_time
        self.bc_coeff = config.bc_coeff
        self.use_bc_loss = config.use_bc_loss

    def collect_rollout(
        self,
        actor_model: ReCogDriveDiffusionPlanner,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        tokens_list: List[str],
    ) -> Dict[str, Any]:
        if self.ref_model.device != actor_model.device:
            self.ref_model.to(actor_model.device)

        B = vl_features.shape[0]
        G = self.sample_time

        vl_features_rep = vl_features.repeat_interleave(G, 0)
        his_traj_rep = action_input.his_traj.repeat_interleave(G, 0)
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)

        with torch.no_grad():
            chains, trajs = actor_model.sample_chain(
                vl_features_rep, his_traj_rep, status_feature_rep, deterministic=False
            )

            tokens_rep = [tok for tok in tokens_list for _ in range(G)]
            rewards = self._reward_fn(trajs, tokens_rep)

            reward_mean = rewards.mean()
            reward_std = rewards.std(unbiased=False)
            reward_nonzero_frac = (rewards != 0).float().mean()
            reward_p10 = torch.quantile(rewards, 0.10)
            reward_p50 = torch.quantile(rewards, 0.50)
            reward_p90 = torch.quantile(rewards, 0.90)

            rewards_matrix = rewards.view(B, G)
            mean_r = rewards_matrix.mean(dim=1, keepdim=True)
            std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
            advantages = ((rewards_matrix - mean_r) / std_r).view(-1)

            adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
            adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
            advantages = advantages.clamp(min=adv_min, max=adv_max)

            num_denoising_steps = chains.shape[1] - 1
            denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
            discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))

            adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)
            discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)
            adv_weighted_flat = (adv_steps * discount).reshape(-1).detach()

            old_log_probs = actor_model.get_logprobs(
                vl_features_rep, his_traj_rep, status_feature_rep, chains, deterministic=False
            )
            old_log_probs = old_log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])

            denoising_steps = chains.shape[1] - 1
            old_log_probs = old_log_probs.view(B * G, denoising_steps).mean(dim=1).detach()
            adv_weighted_flat = adv_weighted_flat.view(B * G, denoising_steps).mean(dim=1).detach()

            teacher_chains = None
            if self.use_bc_loss and self.bc_coeff > 0:
                teacher_chains, _ = self.ref_model.sample_chain(
                    vl_features,
                    action_input.his_traj,
                    action_input.status_feature,
                    deterministic=False,
                )
                teacher_chains = teacher_chains.detach()

        sample_to_base = torch.arange(B, device=vl_features.device).repeat_interleave(G)

        return {
            "B": B,
            "G": G,
            "num_samples": B * G,
            "vl_rep": vl_features_rep.detach(),
            "his_rep": his_traj_rep.detach(),
            "status_rep": status_feature_rep.detach(),
            "chains": chains.detach(),
            "old_log_probs": old_log_probs,
            "advantages": adv_weighted_flat,
            "rewards": rewards.detach(),
            "sample_to_base": sample_to_base,
            "vl_base": vl_features.detach(),
            "his_base": action_input.his_traj.detach(),
            "status_base": action_input.status_feature.detach(),
            "teacher_chains": teacher_chains,
            "reward_mean": reward_mean.detach(),
            "reward_std": reward_std.detach(),
            "reward_nonzero_frac": reward_nonzero_frac.detach(),
            "reward_p10": reward_p10.detach(),
            "reward_p50": reward_p50.detach(),
            "reward_p90": reward_p90.detach(),
            "missing_unique_token_frac": torch.tensor(
                float(getattr(self, "_last_reward_debug", {}).get("missing_unique_token_frac", 0.0)),
                device=vl_features.device,
            ),
            "reward_error_count": torch.tensor(
                float(getattr(self, "_last_reward_debug", {}).get("reward_error_count", 0)),
                device=vl_features.device,
            ),
        }

    def compute_minibatch_loss(
        self,
        actor_model: ReCogDriveDiffusionPlanner,
        rollout: Dict[str, Any],
        mb_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        vl_mb = rollout["vl_rep"][mb_idx]
        his_mb = rollout["his_rep"][mb_idx]
        status_mb = rollout["status_rep"][mb_idx]
        chains_mb = rollout["chains"][mb_idx]
        old_logp_mb = rollout["old_log_probs"][mb_idx]
        adv_mb = rollout["advantages"][mb_idx]

        new_log_probs = actor_model.get_logprobs(
            vl_mb,
            his_mb,
            status_mb,
            chains_mb,
            deterministic=False,
        )
        new_log_probs = new_log_probs.clamp(min=-5, max=2).mean(dim=[1, 2])
        denoising_steps = chains_mb.shape[1] - 1
        new_log_probs = new_log_probs.view(-1, denoising_steps).mean(dim=1)

        ratio = torch.exp(new_log_probs - old_logp_mb)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)

        surr1 = ratio * adv_mb
        surr2 = ratio_clipped * adv_mb
        policy_loss = -torch.min(surr1, surr2).mean()

        bc_loss = torch.tensor(0.0, device=policy_loss.device)
        teacher_chains = rollout["teacher_chains"]
        if teacher_chains is not None and self.use_bc_loss and self.bc_coeff > 0:
            base_ids = torch.unique(rollout["sample_to_base"][mb_idx])
            teacher_mb = teacher_chains[base_ids]
            bc_logp = actor_model.get_logprobs(
                rollout["vl_base"][base_ids],
                rollout["his_base"][base_ids],
                rollout["status_base"][base_ids],
                teacher_mb,
                deterministic=False,
            )
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = teacher_mb.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, teacher_mb.shape[2], teacher_mb.shape[3]).mean(dim=[1, 2, 3])
            bc_loss = -bc_logp.mean()

        total_loss = policy_loss + self.bc_coeff * bc_loss

        approx_kl = (old_logp_mb - new_log_probs).mean().detach()
        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().detach()

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
    



class ReinforcePlusPlusAlgorithm(ReinforceAlgorithm):
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