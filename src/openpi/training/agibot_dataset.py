"""Native reader for the CogACT AGIBot episodic datasets.

The reader intentionally ports only the data semantics required by the CogACT
baseline.  It does not depend on the CogACT package at runtime.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
import dataclasses
import json
import logging
from pathlib import Path
from typing import Literal, SupportsIndex

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VIEW_NAMES = ("head_color", "hand_left_color", "hand_right_color")
_ACTION_DIM = 32
_COMPACT_ACTION_DIM = 20


@dataclasses.dataclass(frozen=True)
class AgibotDatasetSpec:
    """One CogACT dataset directory inside ``dataset_root``."""

    dataset_folder: str
    metadata_file: str
    weight: float = 1.0


@dataclasses.dataclass(frozen=True)
class AgibotDatasetConfig:
    """Configuration for the native AGIBot reader."""

    dataset_root: str
    datasets: tuple[AgibotDatasetSpec, ...]
    action_source: str = "actions_cmd"
    quality_allowlist: tuple[str, ...] = ("good", "medium")
    stride: int = 1
    image_drop_strategy: Literal["none", "balanced_5way"] = "balanced_5way"
    extrinsic_index_mode: Literal["global", "legacy_local"] = "legacy_local"
    terminal_repeat_valid_steps: int = 3


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternions to rotation matrices."""

    quaternion = np.asarray(quaternion, dtype=np.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternion with final dimension 4, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Cannot convert a zero-norm quaternion")
    x, y, z, w = np.moveaxis(quaternion / norm, -1, 0)

    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Match SciPy's ``Rotation.from_euler('xyz', euler)`` convention."""

    euler = np.asarray(euler, dtype=np.float32)
    if euler.shape[-1] != 3:
        raise ValueError(f"Expected xyz Euler angles with final dimension 3, got {euler.shape}")
    x, y, z = np.moveaxis(euler, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)

    matrix = np.empty((*euler.shape[:-1], 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = cz * cy
    matrix[..., 0, 1] = cz * sy * sx - sz * cx
    matrix[..., 0, 2] = cz * sy * cx + sz * sx
    matrix[..., 1, 0] = sz * cy
    matrix[..., 1, 1] = sz * sy * sx + cz * cx
    matrix[..., 1, 2] = sz * sy * cx - cz * sx
    matrix[..., 2, 0] = -sy
    matrix[..., 2, 1] = cy * sx
    matrix[..., 2, 2] = cy * cx
    return matrix


def matrix_to_cogact_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode the first two *rows*, matching CogACT's actual implementation."""

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices, got {matrix.shape}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def cogact_rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Decode CogACT's row-based 6D rotations with Gram-Schmidt."""

    rotation_6d = np.asarray(rotation_6d, dtype=np.float32)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotations, got {rotation_6d.shape}")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _compact_eef_feature(position: np.ndarray, rotation_6d: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Pack bilateral EEF data in CogACT sparse-slot order."""

    return np.concatenate(
        (
            position[..., :3],
            gripper[..., :1],
            rotation_6d[..., :6],
            position[..., 3:6],
            gripper[..., 1:2],
            rotation_6d[..., 6:12],
        ),
        axis=-1,
        dtype=np.float32,
    )


class _AgibotSubDataset:
    def __init__(
        self,
        dataset_root: Path,
        spec: AgibotDatasetSpec,
        config: AgibotDatasetConfig,
        action_horizon: int,
        *,
        load_images: bool,
    ):
        self.dataset_dir = dataset_root / spec.dataset_folder
        self.spec = spec
        self.config = config
        self.action_horizon = action_horizon
        self.load_images = load_images

        if config.stride != 1:
            raise NotImplementedError("The CogACT baseline currently requires stride=1")
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if config.terminal_repeat_valid_steps < 0:
            raise ValueError("terminal_repeat_valid_steps must be non-negative")

        metadata_path = self.dataset_dir / spec.metadata_file
        action_dir = self.dataset_dir / config.action_source
        action_metadata_path = action_dir / "meta_data.json"
        camera_path = action_dir / "camera_param.npy"
        for path in (metadata_path, action_metadata_path, camera_path):
            if not path.exists():
                raise FileNotFoundError(path)

        metadata = np.load(metadata_path, allow_pickle=True).item()
        start_end = np.asarray(metadata["start_end"], dtype=np.int64)
        quality = np.asarray(metadata.get("quality", ["good"] * len(start_end)))
        keep = np.isin(quality, config.quality_allowlist)

        self.video_paths = np.asarray(metadata["video_path"])[keep]
        self.instructions = np.asarray(metadata["instructions"])[keep]
        self.start_end = start_end[keep]
        self.lengths = self.start_end[:, 1] - self.start_end[:, 0] - 1
        self.episode_finished = np.asarray(
            metadata.get("episode_finished", np.ones(len(start_end), dtype=bool)), dtype=bool
        )[keep]
        if np.any(self.lengths <= 0):
            raise ValueError(f"Dataset {self.dataset_dir} contains non-positive episode lengths")

        self._episode_offsets = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
        self._action_dir = action_dir
        with action_metadata_path.open() as file:
            self._action_metadata = json.load(file)
        camera_data = np.load(camera_path, allow_pickle=True).item()
        self._camera_extrinsics = camera_data["camera2extrinsic"]

        self._action_cache: dict[str, np.memmap] = {}
        self._lmdb_env = None
        self._lmdb_path = self.dataset_dir / "lmdb" / "frames.lmdb"
        if load_images and not self._lmdb_path.exists():
            raise FileNotFoundError(self._lmdb_path)

        logger.info(
            "Loaded AGIBot dataset %s: %d/%d episodes, %d frame samples",
            spec.dataset_folder,
            len(self.lengths),
            len(start_end),
            len(self),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_action_cache"] = {}
        state["_lmdb_env"] = None
        return state

    def __len__(self) -> int:
        return int(self._episode_offsets[-1])

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_id = bisect.bisect_right(self._episode_offsets, index) - 1
        local_frame = index - int(self._episode_offsets[episode_id])
        return episode_id, local_frame

    def _get_action_array(self, video_path: str) -> np.memmap:
        if video_path not in self._action_cache:
            try:
                task_metadata = self._action_metadata[video_path]
            except KeyError as error:
                raise KeyError(f"Missing action metadata for {video_path}") from error
            action_dim = int(sum(task_metadata["dim_list"]))
            if action_dim != 56:
                raise ValueError(f"Expected 56D AGIBot action records for {video_path}, got {action_dim}")
            action_path = self._action_dir / video_path / "action.npy"
            if not action_path.exists():
                raise FileNotFoundError(action_path)
            self._action_cache[video_path] = np.memmap(
                action_path,
                dtype=np.float32,
                mode="r",
                shape=(int(task_metadata["length"]), action_dim),
            )
        return self._action_cache[video_path]

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            try:
                import lmdb
            except ImportError as error:
                raise ImportError("Reading AGIBot images requires the 'lmdb' package") from error
            self._lmdb_env = lmdb.open(
                str(self._lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._lmdb_env

    def _read_image(self, video_path: str, view_name: str, global_frame: int) -> np.ndarray:
        key = f"{video_path}:{view_name}:{global_frame:05d}".encode()
        with self._get_lmdb_env().begin(write=False) as transaction:
            encoded = transaction.get(key)
        if encoded is None:
            raise KeyError(f"Frame not found in LMDB: {key.decode()}")
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {key.decode()}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preserve the task-specific head crop in CogACT's AGIBot loader.
        if view_name == "head_color" and video_path.split("/", 1)[0] == "3173":
            if image.shape[0] == 800:
                image = image[40:760]
            elif image.shape[0] == 500:
                image = image[25:475]
        return image

    def _get_cam2world(self, video_path: str, frame_index: int) -> np.ndarray:
        key = f"{video_path}_head"
        if key not in self._camera_extrinsics:
            raise KeyError(f"Missing head camera extrinsics for {video_path}")
        params = np.asarray(self._camera_extrinsics[key], dtype=np.float32)
        index = 0 if len(params) == 1 else frame_index
        if not 0 <= index < len(params):
            raise IndexError(f"Extrinsic index {index} is outside [0, {len(params)}) for {video_path}")

        cam2world = np.eye(4, dtype=np.float32)
        cam2world[:3, 3] = params[index, :3]
        if params.shape[1] == 7:
            cam2world[:3, :3] = quaternion_xyzw_to_matrix(params[index, 3:7])
        elif params.shape[1] == 6:
            cam2world[:3, :3] = euler_xyz_to_matrix(params[index, 3:6])
        else:
            raise ValueError(f"Expected 6D or 7D camera parameters for {key}, got {params.shape}")
        return cam2world

    def __getitem__(self, index: SupportsIndex) -> dict:
        episode_id, local_frame = self._resolve_index(index.__index__())
        episode_start, episode_end = self.start_end[episode_id]
        global_frame = int(episode_start + local_frame)
        video_path = str(self.video_paths[episode_id])

        unbounded_indices = global_frame + np.arange(self.action_horizon + 1, dtype=np.int64) * self.config.stride
        state_indices = np.clip(unbounded_indices, global_frame, episode_end - 1)
        valid_repeat_steps = self.config.terminal_repeat_valid_steps if self.episode_finished[episode_id] else 0
        action_time_mask = unbounded_indices[1:] < episode_end + valid_repeat_steps

        action_array = self._get_action_array(video_path)
        if episode_end > len(action_array):
            raise IndexError(f"Episode end {episode_end} exceeds action length {len(action_array)} for {video_path}")
        state_records = np.asarray(action_array[state_indices])
        state_quaternions = state_records[:, 22:30]
        state_positions = state_records[:, 30:36]
        state_grippers = state_records[:, 54:56] / 120.0

        left_matrices = quaternion_xyzw_to_matrix(state_quaternions[:, :4])
        right_matrices = quaternion_xyzw_to_matrix(state_quaternions[:, 4:8])
        rotation_6d = np.concatenate(
            (matrix_to_cogact_rotation_6d(left_matrices), matrix_to_cogact_rotation_6d(right_matrices)), axis=-1
        )

        current_eef_state = _compact_eef_feature(state_positions[0], rotation_6d[0], state_grippers[0])
        extrinsic_frame = local_frame if self.config.extrinsic_index_mode == "legacy_local" else global_frame
        cam2world = self._get_cam2world(video_path, extrinsic_frame)
        camera_state = np.concatenate((cam2world[:3, 3], cam2world[:3, :3].reshape(9)), dtype=np.float32)
        state = np.concatenate((current_eef_state, camera_state), dtype=np.float32)

        # CogACT uses future observed EEF poses but current-step actions_cmd gripper commands.
        command_grippers = np.asarray(action_array[state_indices[:-1], 20:22])
        compact_actions = _compact_eef_feature(state_positions[1:], rotation_6d[1:], command_grippers)
        actions = np.zeros((self.action_horizon, _ACTION_DIM), dtype=np.float32)
        actions[:, :_COMPACT_ACTION_DIM] = compact_actions
        action_loss_mask = np.zeros_like(actions, dtype=bool)
        action_loss_mask[:, :_COMPACT_ACTION_DIM] = action_time_mask[:, None]

        sample = {
            "state": state,
            "actions": actions,
            "action_loss_mask": action_loss_mask,
            "prompt": str(self.instructions[episode_id]).strip().lower(),
        }
        if self.load_images:
            images = {name: self._read_image(video_path, name, global_frame) for name in _VIEW_NAMES}
            image_masks = dict.fromkeys(_VIEW_NAMES, np.True_)
            if self.config.image_drop_strategy == "balanced_5way":
                drop_cases = (
                    (),
                    ("hand_left_color", "hand_right_color"),
                    ("head_color",),
                    ("hand_left_color",),
                    ("hand_right_color",),
                )
                for name in drop_cases[np.random.randint(len(drop_cases))]:
                    images[name] = np.zeros_like(images[name])
                    image_masks[name] = np.False_
            sample["images"] = images
            sample["image_masks"] = image_masks
        return sample


class AgibotDataset:
    """Concatenate AGIBot datasets using CogACT's weight=1 sampling semantics."""

    def __init__(self, config: AgibotDatasetConfig, action_horizon: int, *, load_images: bool = True):
        if not config.datasets:
            raise ValueError("At least one AGIBot dataset must be configured")
        if any(spec.weight != 1.0 for spec in config.datasets):
            raise NotImplementedError("Native AGIBot sampling currently supports only weight=1.0")
        root = Path(config.dataset_root)
        self.datasets = [
            _AgibotSubDataset(root, spec, config, action_horizon, load_images=load_images) for spec in config.datasets
        ]
        self._dataset_offsets = np.concatenate(
            ([0], np.cumsum([len(dataset) for dataset in self.datasets], dtype=np.int64))
        )

    def __len__(self) -> int:
        return int(self._dataset_offsets[-1])

    @property
    def dataset_lengths(self) -> tuple[int, ...]:
        return tuple(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        dataset_id = bisect.bisect_right(self._dataset_offsets, index) - 1
        local_index = index - int(self._dataset_offsets[dataset_id])
        return self.datasets[dataset_id][local_index]


def load_agibot_dataset_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | None = None,
) -> AgibotDatasetConfig:
    """Load the dataset root and arbitrary list of sub-datasets from a JSON manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"AGIBot dataset manifest not found: {manifest_path}")
    with manifest_path.open() as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"AGIBot dataset manifest must be a JSON object: {manifest_path}")

    unknown_keys = set(manifest) - {"dataset_root", "datasets"}
    if unknown_keys:
        raise ValueError(f"Unknown AGIBot dataset manifest keys: {sorted(unknown_keys)}")
    root = dataset_root or manifest.get("dataset_root")
    if not root:
        raise ValueError("AGIBot dataset manifest must define dataset_root, or dataset_root must be overridden")
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = manifest_path.parent / root_path

    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("AGIBot dataset manifest must contain a non-empty datasets list")
    allowed_entry_keys = {"dataset_folder", "metadata_file", "weight"}
    datasets = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"AGIBot dataset manifest entry {index} must be a JSON object")
        unknown_entry_keys = set(entry) - allowed_entry_keys
        if unknown_entry_keys:
            raise ValueError(f"Unknown keys in AGIBot dataset manifest entry {index}: {sorted(unknown_entry_keys)}")
        try:
            datasets.append(
                AgibotDatasetSpec(
                    dataset_folder=str(entry["dataset_folder"]),
                    metadata_file=str(entry["metadata_file"]),
                    weight=float(entry.get("weight", 1.0)),
                )
            )
        except KeyError as error:
            raise ValueError(f"AGIBot dataset manifest entry {index} is missing {error.args[0]!r}") from error

    return AgibotDatasetConfig(dataset_root=str(root_path), datasets=tuple(datasets))


def dataset_sample_ratios(lengths: Sequence[int]) -> np.ndarray:
    """Compute expected sample ratios for weight=1 concatenation."""

    lengths = np.asarray(lengths, dtype=np.float64)
    return lengths / lengths.sum()
