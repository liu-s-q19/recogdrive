import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    StateIndex,
    WeightedMetricIndex,
)


class _EmptyOccupancyMap:
    tokens = []

    def query(self, *_args, **_kwargs):
        return ()


class _BoundedObservation:
    def __init__(self, max_valid_idx: int) -> None:
        self.max_valid_idx = max_valid_idx
        self.requested_indices = []
        self.collided_track_ids = []
        self.red_light_token = "red_light"
        self.unique_objects = {}

    def __getitem__(self, time_idx: int) -> _EmptyOccupancyMap:
        self.requested_indices.append(time_idx)
        assert 0 <= time_idx <= self.max_valid_idx, f"PDMObservation: index {time_idx} out of range!"
        return _EmptyOccupancyMap()


def _build_ego_coords(num_steps: int) -> np.ndarray:
    coords = np.zeros((1, num_steps, len(BBCoordsIndex), 2), dtype=np.float64)
    for time_idx in range(num_steps):
        x = float(time_idx)
        coords[0, time_idx, BBCoordsIndex.FRONT_LEFT] = [x + 2.0, 1.0]
        coords[0, time_idx, BBCoordsIndex.REAR_LEFT] = [x, 1.0]
        coords[0, time_idx, BBCoordsIndex.REAR_RIGHT] = [x, -1.0]
        coords[0, time_idx, BBCoordsIndex.FRONT_RIGHT] = [x + 2.0, -1.0]
        coords[0, time_idx, BBCoordsIndex.CENTER] = [x + 1.0, 0.0]
    return coords


def test_calculate_ttc_respects_observation_horizon_boundary() -> None:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    scorer = PDMScorer(proposal_sampling=proposal_sampling)
    observation = _BoundedObservation(max_valid_idx=40)

    num_steps = proposal_sampling.num_poses + 1
    scorer._num_proposals = 1
    scorer._observation = observation
    scorer._states = np.zeros((1, num_steps, StateIndex.size()), dtype=np.float64)
    scorer._ego_coords = _build_ego_coords(num_steps)
    scorer._ego_areas = np.zeros((1, num_steps, len(EgoAreaIndex)), dtype=np.bool_)
    scorer._weighted_metrics = np.zeros((len(WeightedMetricIndex), 1), dtype=np.float64)
    scorer._ttc_time_idcs = np.full(1, np.inf, dtype=np.float64)
    scorer._drivable_area_map = None

    scorer._calculate_ttc()

    assert max(observation.requested_indices) == 40
    assert scorer._weighted_metrics[WeightedMetricIndex.TTC, 0] == 1.0
