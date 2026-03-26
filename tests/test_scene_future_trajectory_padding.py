from types import SimpleNamespace

import numpy as np

from navsim.common.dataclasses import (
    Annotations,
    Camera,
    Cameras,
    EgoStatus,
    Frame,
    Lidar,
    Scene,
    SceneMetadata,
)


def _empty_cameras() -> Cameras:
    camera = Camera()
    return Cameras(
        cam_f0=camera,
        cam_l0=camera,
        cam_l1=camera,
        cam_l2=camera,
        cam_r0=camera,
        cam_r1=camera,
        cam_r2=camera,
        cam_b0=camera,
    )


def _empty_annotations() -> Annotations:
    return Annotations(
        boxes=np.zeros((0, 7), dtype=np.float32),
        names=[],
        velocity_3d=np.zeros((0, 3), dtype=np.float32),
        instance_tokens=[],
        track_tokens=[],
    )


def test_scene_future_trajectory_pads_with_last_available_pose() -> None:
    frames = []
    for idx in range(6):
        frames.append(
            Frame(
                token=f"token-{idx}",
                timestamp=idx,
                roadblock_ids=[],
                traffic_lights=[],
                annotations=_empty_annotations(),
                ego_status=EgoStatus(
                    ego_pose=np.array([float(idx), 0.0, 0.0], dtype=np.float64),
                    ego_velocity=np.zeros(2, dtype=np.float32),
                    ego_acceleration=np.zeros(2, dtype=np.float32),
                    driving_command=np.zeros(1, dtype=np.int64),
                    in_global_frame=True,
                ),
                lidar=Lidar(),
                cameras=_empty_cameras(),
            )
        )

    scene = Scene(
        scene_metadata=SceneMetadata(
            log_name="log",
            scene_token="scene",
            map_name="us-nv-las-vegas-strip",
            initial_token="token-3",
            num_history_frames=4,
            num_future_frames=8,
        ),
        map_api=SimpleNamespace(),
        frames=frames,
    )

    trajectory = scene.get_future_trajectory(num_trajectory_frames=8)

    assert trajectory.poses.shape == (8, 3)
    np.testing.assert_allclose(trajectory.poses[:2], np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    np.testing.assert_allclose(trajectory.poses[2:], np.repeat([[2.0, 0.0, 0.0]], 6, axis=0))
