from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent


def test_recogdrive_single_camera_sensor_config_skips_lidar() -> None:
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        cam_type="single",
        cache_hidden_state=True,
        cache_mode=True,
        vlm_type="internvl",
        vlm_path="/tmp/mock-vlm",
    )

    sensor_config = agent.get_sensor_config()

    assert sensor_config.cam_f0 == [0, 1, 2, 3]
    assert sensor_config.lidar_pc is False
