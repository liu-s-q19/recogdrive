from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from tqdm import tqdm
import pickle
import lzma

from navsim.common.cache_metadata import ensure_v2_cache_metadata, V2_CACHE_SCHEMA_VERSION
from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache

FrameList = List[Dict[str, Any]]


def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Tuple[Dict[str, FrameList], List[str]]:
    """
    Load a set of scenes from dataset, while applying scene filter configuration.
    :param data_path: root directory of log folder
    :param scene_filter: scene filtering configuration class
    :return: dictionary of raw logs format, and final-frame tokens used to match second-stage synthetic scenes
    """

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """Helper function to split frame list according to sampling specification."""
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    filtered_scenes: Dict[str, Scene] = {}
    final_frame_tokens: List[str] = []
    stop_loading: bool = False

    # filter logs
    log_files = list(data_path.iterdir())
    if scene_filter.log_names is not None:
        log_files = [log_file for log_file in log_files if log_file.name.replace(".pkl", "") in scene_filter.log_names]

    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)
    else:
        filter_tokens = False

    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            # Filter scenes which are too short
            if len(frame_list) < scene_filter.num_frames:
                continue

            # Filter scenes with no route
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # Filter by token
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue

            filtered_scenes[token] = frame_list
            final_frame_tokens.append(frame_list[scene_filter.num_frames - 1]["token"])

            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        if stop_loading:
            break

    return filtered_scenes, final_frame_tokens


def filter_synthetic_scenes(
    data_path: Path, scene_filter: SceneFilter, stage1_scenes_final_frame_tokens: List[str]
) -> Dict[str, Tuple[Path, str]]:
    """Load synthetic scenes associated with the already selected original scenes."""
    loaded_scenes: Dict[str, Tuple[Path, str]] = {}
    synthetic_scenes_paths = list(data_path.iterdir())

    filter_logs = scene_filter.log_names is not None
    filter_tokens = scene_filter.synthetic_scene_tokens is not None

    for scene_path in tqdm(synthetic_scenes_paths, desc="Loading synthetic scenes"):
        synthetic_scene = Scene.load_from_disk(scene_path, None, None)

        if filter_tokens and synthetic_scene.scene_metadata.initial_token not in scene_filter.synthetic_scene_tokens:
            continue

        log_name = synthetic_scene.scene_metadata.log_name
        if filter_logs and log_name not in scene_filter.log_names:
            continue

        if (
            not filter_tokens
            and synthetic_scene.scene_metadata.corresponding_original_scene not in stage1_scenes_final_frame_tokens
        ):
            continue

        loaded_scenes[synthetic_scene.scene_metadata.initial_token] = (scene_path, log_name)

    return loaded_scenes


class SceneLoader:
    """Simple data loader of scenes from logs."""

    def __init__(
        self,
        data_path: Path,
        scene_filter: SceneFilter,
        original_sensor_path: Optional[Path] = None,
        synthetic_sensor_path: Optional[Path] = None,
        synthetic_scenes_path: Optional[Path] = None,
        sensor_blobs_path: Optional[Path] = None,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        load_image_path: bool = False,
    ):
        """
        Initializes the scene data loader.
        :param data_path: root directory of log folder
        :param original_sensor_path: root directory of original sensor data
        :param synthetic_sensor_path: root directory of synthetic sensor data
        :param synthetic_scenes_path: root directory of serialized synthetic scenes
        :param sensor_blobs_path: legacy alias for original sensor data
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        """

        if original_sensor_path is None:
            original_sensor_path = sensor_blobs_path

        self.scene_frames_dicts, stage1_scenes_final_frame_tokens = filter_scenes(data_path, scene_filter)
        self._original_sensor_path = original_sensor_path
        self._synthetic_sensor_path = synthetic_sensor_path
        self._scene_filter = scene_filter
        self._sensor_config = sensor_config
        self.load_image_path = load_image_path

        if scene_filter.include_synthetic_scenes:
            assert (
                synthetic_scenes_path is not None
            ), "Synthetic scenes path cannot be None when include_synthetic_scenes is enabled."
            self.synthetic_scenes = filter_synthetic_scenes(
                data_path=synthetic_scenes_path,
                scene_filter=scene_filter,
                stage1_scenes_final_frame_tokens=stage1_scenes_final_frame_tokens,
            )
            self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())
        else:
            self.synthetic_scenes = {}
            self.synthetic_scenes_tokens = set()

        self.token_to_log_file: Dict[str, str] = {}
        self._build_token_to_log_file()

    def _build_token_to_log_file(self):
        """Builds the token_to_log_file dictionary."""
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]
            self.token_to_log_file[token] = log_name
        for token, (_, log_name) in self.synthetic_scenes.items():
            self.token_to_log_file[token] = log_name

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.scene_frames_dicts.keys()) + list(self.synthetic_scenes.keys())

    @property
    def tokens_stage_one(self) -> List[str]:
        return list(self.scene_frames_dicts.keys())

    @property
    def reactive_tokens_stage_two(self) -> Optional[List[str]]:
        reactive_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        if reactive_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(reactive_tokens))

    @property
    def non_reactive_tokens_stage_two(self) -> Optional[List[str]]:
        non_reactive_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        if non_reactive_tokens is None:
            return None
        return list(set(self.synthetic_scenes_tokens) & set(non_reactive_tokens))

    def __len__(self) -> int:
        """
        :return: number for scenes possible to load.
        """
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        :param idx: index of scene
        :return: unique scene identifier
        """
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._synthetic_sensor_path,
                sensor_config=self._sensor_config,
                lidar_sensor_blobs_path=self._original_sensor_path,
                load_image_path=self.load_image_path,
            )
        return Scene.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._original_sensor_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
            load_image_path=self.load_image_path,
        )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        assert token in self.tokens
        if token in self.synthetic_scenes:
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],
                sensor_blobs_path=self._synthetic_sensor_path,
                sensor_config=self._sensor_config,
                lidar_sensor_blobs_path=self._original_sensor_path,
                load_image_path=self.load_image_path,
            ).get_agent_input()
        return AgentInput.from_scene_dict_list(
            self.scene_frames_dicts[token],
            self._original_sensor_path,
            num_history_frames=self._scene_filter.num_history_frames,
            sensor_config=self._sensor_config,
            load_image_path=self.load_image_path,
        )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each logs file given filtering.
        :return: dictionary of logs names and tokens
        """
        # generate a dict that contains a list of tokens for each log-name
        tokens_per_logs: Dict[str, List[str]] = {}
        for token, log_name in self.token_to_log_file.items():
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)
            else:
                tokens_per_logs[log_name] = [token]
        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(
        self,
        cache_path: Path,
        file_name: str = "metric_cache.pkl",
        require_v2_metadata: bool = False,
        expected_schema_version: str = V2_CACHE_SCHEMA_VERSION,
    ):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self._cache_path = Path(cache_path)
        self._require_v2_metadata = require_v2_metadata
        self._expected_schema_version = expected_schema_version
        self.cache_metadata = (
            ensure_v2_cache_metadata(self._cache_path, expected_schema_version)
            if require_v2_metadata
            else None
        )
        self.metric_cache_paths = self._load_metric_cache_paths(self._cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"

        # Prefer the legacy/indexed format: metadata/*.csv contains cache file paths.
        if metadata_dir.exists():
            csv_files = [file for file in metadata_dir.iterdir() if file.suffix == ".csv"]
            if csv_files:
                metadata_file = csv_files[0]
                with open(str(metadata_file), "r") as f:
                    cache_paths = f.read().splitlines()[1:]
                # Each line is a cache file path; token is assumed to be the parent folder name.
                metric_cache_dict = {Path(p).parts[-2]: Path(p) for p in cache_paths if p}
                if metric_cache_dict:
                    return metric_cache_dict

        # Fallback: scan the cache folder recursively for per-token files.
        # Current workspace layout is like:
        #   metric_cache_train/<log_segment>/unknown/<token>/metric_cache.pkl
        # or sometimes compressed variants.
        metric_cache_dict: Dict[str, Path] = {}
        patterns = ["metric_cache.pkl", "metric_cache.pkl.xz", "metric_cache.pkl.lzma"]
        for pat in patterns:
            for p in cache_path.rglob(pat):
                token = p.parent.name
                metric_cache_dict[token] = p
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        path = self.metric_cache_paths[token]
        try:
            with lzma.open(path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
        except lzma.LZMAError:
            with open(path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

        if self._require_v2_metadata:
            missing_fields = [
                field_name
                for field_name in ("scene_type", "human_trajectory", "past_human_trajectory", "map_parameters")
                if not hasattr(metric_cache, field_name)
            ]
            if missing_fields:
                raise ValueError(
                    f"Legacy metric cache object loaded from {path} is missing v2 fields: {missing_fields}"
                )
            if getattr(metric_cache, "map_parameters", None) is None:
                raise ValueError(f"Legacy metric cache object loaded from {path} is missing map_parameters")
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
