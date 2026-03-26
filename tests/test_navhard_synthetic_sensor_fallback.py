import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig


def _build_sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=True,
        cam_l0=False,
        cam_l1=False,
        cam_l2=False,
        cam_r0=False,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=True,
    )


def _write_synthetic_scene(scene_path: Path, log_name: str, initial_token: str, frame_token: str) -> None:
    scene_data = {
        "scene_metadata": {
            "log_name": log_name,
            "scene_token": initial_token,
            "map_name": "us-nv-las-vegas-strip",
            "initial_token": initial_token,
            "num_history_frames": 1,
            "num_future_frames": 0,
            "corresponding_original_scene": None,
            "corresponding_original_initial_token": None,
        },
        "frames": [
            {
                "token": frame_token,
                "timestamp": 0,
                "roadblock_ids": ["rb-1"],
                "traffic_lights": [],
                "annotations": {
                    "boxes": np.zeros((0, 7), dtype=np.float32),
                    "names": [],
                    "velocity_3d": np.zeros((0, 3), dtype=np.float32),
                    "instance_tokens": [],
                    "track_tokens": [],
                },
                "ego_status": {
                    "ego_pose": np.zeros(3, dtype=np.float64),
                    "ego_velocity": np.zeros(2, dtype=np.float32),
                    "ego_acceleration": np.zeros(2, dtype=np.float32),
                    "driving_command": np.zeros(1, dtype=np.int64),
                    "in_global_frame": True,
                },
                "lidar_path": f"{log_name}/MergedPointCloud/{frame_token}.pcd",
                "camera_dict": {
                    "cam_f0": {
                        "data_path": f"{log_name}/CAM_F0/{frame_token}.jpg",
                        "sensor2lidar_rotation": np.eye(3, dtype=np.float32),
                        "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
                        "cam_intrinsic": np.eye(3, dtype=np.float32),
                        "distortion": np.zeros(5, dtype=np.float32),
                    },
                    "cam_l0": {},
                    "cam_l1": {},
                    "cam_l2": {},
                    "cam_r0": {},
                    "cam_r1": {},
                    "cam_r2": {},
                    "cam_b0": {},
                },
            }
        ],
        "extended_traffic_light_data": None,
        "extended_detections_tracks": None,
    }

    with scene_path.open("wb") as file_obj:
        pickle.dump(scene_data, file_obj, protocol=pickle.HIGHEST_PROTOCOL)


def test_scene_loader_loads_synthetic_camera_from_synthetic_root_and_lidar_from_original_root(
    tmp_path: Path, monkeypatch
) -> None:
    logs_path = tmp_path / "logs"
    logs_path.mkdir()
    original_sensor_path = tmp_path / "original_sensor_blobs"
    synthetic_sensor_path = tmp_path / "synthetic_sensor_blobs"
    synthetic_scenes_path = tmp_path / "synthetic_scene_pickles"
    original_sensor_path.mkdir()
    synthetic_sensor_path.mkdir()
    synthetic_scenes_path.mkdir()

    log_name = "2021.08.30.14.54.34_veh-40_00439_00835"
    token = "f8a62eeaee012f32c"

    camera_path = synthetic_sensor_path / log_name / "CAM_F0" / f"{token}.jpg"
    camera_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(camera_path)

    lidar_path = original_sensor_path / log_name / "MergedPointCloud" / f"{token}.pcd"
    lidar_path.parent.mkdir(parents=True)
    lidar_path.write_bytes(b"pcd")

    _write_synthetic_scene(synthetic_scenes_path / f"{token}.pkl", log_name, token, token)

    monkeypatch.setattr(Scene, "_build_map_api", classmethod(lambda cls, map_name: object()))
    monkeypatch.setattr(
        "navsim.common.dataclasses.LidarPointCloud.from_buffer",
        lambda *_args, **_kwargs: SimpleNamespace(points=np.zeros((6, 1), dtype=np.float32)),
    )

    loader = SceneLoader(
        data_path=logs_path,
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            has_route=False,
            include_synthetic_scenes=True,
            synthetic_scene_tokens=[token],
        ),
        original_sensor_path=original_sensor_path,
        synthetic_sensor_path=synthetic_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=_build_sensor_config(),
    )

    scene = loader.get_scene_from_token(token)

    assert scene.frames[0].cameras.cam_f0.image.shape == (1, 1, 3)
    assert scene.frames[0].lidar.lidar_pc.shape == (6, 1)


def test_scene_loader_can_return_synthetic_camera_paths_for_cache_mode(tmp_path: Path, monkeypatch) -> None:
    logs_path = tmp_path / "logs"
    logs_path.mkdir()
    original_sensor_path = tmp_path / "original_sensor_blobs"
    synthetic_sensor_path = tmp_path / "synthetic_sensor_blobs"
    synthetic_scenes_path = tmp_path / "synthetic_scene_pickles"
    original_sensor_path.mkdir()
    synthetic_sensor_path.mkdir()
    synthetic_scenes_path.mkdir()

    log_name = "2021.08.30.14.54.34_veh-40_00439_00835"
    token = "f8a62eeaee012f32c"

    camera_path = synthetic_sensor_path / log_name / "CAM_F0" / f"{token}.jpg"
    camera_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(camera_path)

    _write_synthetic_scene(synthetic_scenes_path / f"{token}.pkl", log_name, token, token)

    monkeypatch.setattr(Scene, "_build_map_api", classmethod(lambda cls, map_name: object()))

    loader = SceneLoader(
        data_path=logs_path,
        scene_filter=SceneFilter(
            num_history_frames=1,
            num_future_frames=0,
            has_route=False,
            include_synthetic_scenes=True,
            synthetic_scene_tokens=[token],
        ),
        original_sensor_path=original_sensor_path,
        synthetic_sensor_path=synthetic_sensor_path,
        synthetic_scenes_path=synthetic_scenes_path,
        sensor_config=SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        ),
        load_image_path=True,
    )

    scene = loader.get_scene_from_token(token)

    assert scene.frames[0].cameras.cam_f0.image == camera_path
