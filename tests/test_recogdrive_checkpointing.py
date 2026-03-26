from navsim.agents.recogdrive.recogdrive_checkpointing import (
    resolve_reference_policy_checkpoint,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def test_reference_checkpoint_defaults_to_init_checkpoint() -> None:
    assert resolve_reference_policy_checkpoint(
        checkpoint_path="/tmp/init.ckpt",
        reference_policy_checkpoint="",
    ) == "/tmp/init.ckpt"


def test_reference_checkpoint_preserves_explicit_value() -> None:
    assert resolve_reference_policy_checkpoint(
        checkpoint_path="/tmp/init.ckpt",
        reference_policy_checkpoint="/tmp/ref.ckpt",
    ) == "/tmp/ref.ckpt"


def test_grpo_agent_initializes_rl_algorithm_without_load_checkpoint(monkeypatch) -> None:
    from navsim.agents.recogdrive import recogdrive_agent as agent_module

    class DummyPlanner:
        def __init__(self, cfg):
            self.cfg = cfg

        def cuda(self):
            return self

    class DummyAlgo:
        def __init__(self, cfg, action_head):
            self.cfg = cfg
            self.action_head = action_head

    class DummyGrpoCfg:
        def __init__(self):
            self.metric_cache_path = ""
            self.reference_policy_checkpoint = ""
            self.scene_loader_mode = ""

    class DummyCfg:
        def __init__(self):
            self.grpo_cfg = DummyGrpoCfg()
            self.vlm_size = "large"

    monkeypatch.setattr(agent_module, "make_recogdrive_config", lambda *args, **kwargs: DummyCfg())
    monkeypatch.setattr(agent_module, "ReCogDriveDiffusionPlanner", DummyPlanner)
    monkeypatch.setattr(agent_module, "ReinforceAlgorithm", DummyAlgo)
    monkeypatch.setattr(agent_module, "ReinforcePlusPlusAlgorithm", DummyAlgo)

    agent = agent_module.ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        grpo=True,
        rl_algo_type="reinforce_plus_plus",
        checkpoint_path="/tmp/init.ckpt",
        metric_cache_path="/tmp/metric_cache",
        cache_hidden_state=True,
    )

    assert agent.rl_algo is not None
    assert agent.reference_policy_checkpoint == "/tmp/init.ckpt"
    assert agent.rl_algo.cfg.reference_policy_checkpoint == "/tmp/init.ckpt"
