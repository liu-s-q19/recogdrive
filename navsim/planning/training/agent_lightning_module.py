import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple,Any

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()


class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.use_manual_rl_optimization = bool(getattr(agent, "requires_manual_optimization", False))
        if self.use_manual_rl_optimization:
            self.automatic_optimization = False

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)

            loss = predictions.loss
            reward = predictions.reward
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if not self.use_manual_rl_optimization:
            return self._step(batch, "train")

        features, targets, tokens_list = batch
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]

        vl_features, action_inputs = self.agent.build_rl_training_inputs(features, targets)
        rollout = self.agent.rl_algo.collect_rollout(
            actor_model=self.agent.action_head,
            vl_features=vl_features,
            action_input=action_inputs,
            tokens_list=tokens_list,
        )

        num_samples = int(rollout["num_samples"])
        mini_batch_size = int(self.agent.rl_algo.mini_batch_size)
        ppo_epochs = int(self.agent.rl_algo.ppo_epochs)
        max_grad_norm = float(self.agent.rl_algo.max_grad_norm)
        target_kl = self.agent.rl_algo.target_kl

        total_loss_sum = torch.tensor(0.0, device=vl_features.device)
        policy_loss_sum = torch.tensor(0.0, device=vl_features.device)
        bc_loss_sum = torch.tensor(0.0, device=vl_features.device)
        approx_kl_sum = torch.tensor(0.0, device=vl_features.device)
        clip_frac_sum = torch.tensor(0.0, device=vl_features.device)
        step_count = 0
        early_stop = False

        for _ in range(ppo_epochs):
            perm = torch.randperm(num_samples, device=vl_features.device)
            for start in range(0, num_samples, mini_batch_size):
                mb_idx = perm[start:start + mini_batch_size]
                loss_dict = self.agent.rl_algo.compute_minibatch_loss(
                    actor_model=self.agent.action_head,
                    rollout=rollout,
                    mb_idx=mb_idx,
                )

                optimizer.zero_grad()
                self.manual_backward(loss_dict["loss"])
                if max_grad_norm > 0:
                    self.clip_gradients(
                        optimizer,
                        gradient_clip_val=max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )
                optimizer.step()

                total_loss_sum = total_loss_sum + loss_dict["loss"].detach()
                policy_loss_sum = policy_loss_sum + loss_dict["policy_loss"]
                bc_loss_sum = bc_loss_sum + loss_dict["bc_loss"]
                approx_kl_sum = approx_kl_sum + loss_dict["approx_kl"]
                clip_frac_sum = clip_frac_sum + loss_dict["clip_frac"]
                step_count += 1

                if target_kl is not None and loss_dict["approx_kl"].item() > float(target_kl):
                    early_stop = True
                    break
            if early_stop:
                break

        if step_count == 0:
            step_count = 1

        loss = total_loss_sum / step_count
        policy_loss = policy_loss_sum / step_count
        bc_loss = bc_loss_sum / step_count
        approx_kl = approx_kl_sum / step_count
        clip_frac = clip_frac_sum / step_count
        reward = rollout["rewards"].mean()
        reward_std = rollout.get("reward_std", torch.tensor(0.0, device=loss.device))
        reward_nonzero_frac = rollout.get("reward_nonzero_frac", torch.tensor(0.0, device=loss.device))
        reward_p10 = rollout.get("reward_p10", torch.tensor(0.0, device=loss.device))
        reward_p50 = rollout.get("reward_p50", torch.tensor(0.0, device=loss.device))
        reward_p90 = rollout.get("reward_p90", torch.tensor(0.0, device=loss.device))
        missing_unique_token_frac = rollout.get("missing_unique_token_frac", torch.tensor(0.0, device=loss.device))
        reward_error_count = rollout.get("reward_error_count", torch.tensor(0.0, device=loss.device))

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/reward_std", reward_std, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_nonzero_frac", reward_nonzero_frac, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_p10", reward_p10, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_p50", reward_p50, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_p90", reward_p90, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/missing_unique_token_frac", missing_unique_token_frac, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_error_count", reward_error_count, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/approx_kl", approx_kl, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/clip_frac", clip_frac, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/ppo_update_steps", torch.tensor(float(step_count), device=loss.device), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def on_train_epoch_end(self) -> None:
        if not self.use_manual_rl_optimization:
            return
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, list):
            for sch in scheduler:
                sch.step()
        else:
            scheduler.step()