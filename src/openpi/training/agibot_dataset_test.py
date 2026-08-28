import json
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.policies import agibot_policy
from openpi.training import agibot_dataset as agibot
from openpi.training import config as training_config
from openpi.training import data_loader
import openpi.transforms as transforms


def _write_synthetic_dataset(root: Path, *, terminal: bool = True) -> agibot.AgibotDatasetConfig:
    dataset_dir = root / "synthetic"
    action_dir = dataset_dir / "actions_cmd"
    video_path = "42/episode_0"
    (action_dir / video_path).mkdir(parents=True)

    # The episode starts at a nonzero global index and provides three samples:
    # global frames 2, 3, and 4. Frame 5 is the final real pose target.
    metadata = {
        "start_end": np.array([[2, 6]], dtype=np.int64),
        "quality": np.array(["good"]),
        "video_path": np.array([video_path]),
        "instructions": np.array(["  Pick The Item  "]),
        "episode_finished": np.array([terminal]),
    }
    np.save(dataset_dir / "metadata.npy", metadata, allow_pickle=True)

    records = np.zeros((8, 56), dtype=np.float32)
    records[:, 20] = np.arange(8) + 100
    records[:, 21] = np.arange(8) + 200
    # Identity xyzw quaternions for both EEFs.
    records[:, 25] = 1
    records[:, 29] = 1
    for frame in range(len(records)):
        records[frame, 30:36] = np.arange(6) + frame * 10
    records[:, 54] = 60
    records[:, 55] = 120
    records.tofile(action_dir / video_path / "action.npy")
    (action_dir / "meta_data.json").write_text(json.dumps({video_path: {"dim_list": [56], "length": len(records)}}))

    # Distinct translation at every global frame makes local/global indexing observable.
    camera_params = np.zeros((8, 6), dtype=np.float32)
    camera_params[:, 0] = np.arange(8) + 0.25
    camera_params[:, 1] = np.arange(8) + 0.5
    camera_params[:, 2] = np.arange(8) + 0.75
    np.save(action_dir / "camera_param.npy", {"camera2extrinsic": {f"{video_path}_head": camera_params}})

    lmdb_path = dataset_dir / "lmdb" / "frames.lmdb"
    lmdb_path.mkdir(parents=True)
    environment = lmdb.open(str(lmdb_path), map_size=1 << 20)
    with environment.begin(write=True) as transaction:
        for frame in range(2, 5):
            for view_index, view_name in enumerate(agibot._VIEW_NAMES):  # noqa: SLF001
                image = np.full((12, 16, 3), 20 * (view_index + 1), dtype=np.uint8)
                success, encoded = cv2.imencode(".jpg", image)
                assert success
                key = f"{video_path}:{view_name}:{frame:05d}".encode()
                transaction.put(key, encoded.tobytes())
    environment.close()

    return agibot.AgibotDatasetConfig(
        dataset_root=str(root),
        datasets=(agibot.AgibotDatasetSpec("synthetic", "metadata.npy"),),
        image_drop_strategy="balanced_5way",
        extrinsic_index_mode="global",
    )


def test_dataset_manifest(tmp_path):
    manifest = tmp_path / "datasets.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_root": "relative_data",
                "datasets": [
                    {
                        "dataset_folder": "dataset_a",
                        "metadata_file": "metadata_a.npy",
                    },
                    {
                        "dataset_folder": "dataset_b",
                        "metadata_file": "metadata_b.npy",
                        "weight": 1.0,
                    },
                ],
            }
        )
    )

    config = agibot.load_agibot_dataset_manifest(manifest)
    assert config.dataset_root == str(tmp_path / "relative_data")
    assert [spec.dataset_folder for spec in config.datasets] == ["dataset_a", "dataset_b"]
    assert all(spec.weight == 1.0 for spec in config.datasets)
    assert config.extrinsic_index_mode == "legacy_local"


def test_rotation_conversions_are_row_based():
    quaternion = np.array([0.2, -0.3, 0.1, 0.9], dtype=np.float32)
    matrix = agibot.quaternion_xyzw_to_matrix(quaternion)
    rotation_6d = agibot.matrix_to_cogact_rotation_6d(matrix)

    np.testing.assert_allclose(rotation_6d, matrix[:2].reshape(6), atol=1e-6)
    np.testing.assert_allclose(agibot.cogact_rotation_6d_to_matrix(rotation_6d), matrix, atol=1e-6)

    # xyz Euler rotations and equivalent scalar-last quaternions agree for axis rotations.
    half_angle = np.pi / 4
    for axis in range(3):
        euler = np.zeros(3, dtype=np.float32)
        euler[axis] = np.pi / 2
        quaternion = np.zeros(4, dtype=np.float32)
        quaternion[axis] = np.sin(half_angle)
        quaternion[3] = np.cos(half_angle)
        np.testing.assert_allclose(
            agibot.euler_xyz_to_matrix(euler), agibot.quaternion_xyzw_to_matrix(quaternion), atol=1e-6
        )


def test_synthetic_reader_uses_global_extrinsics_and_action_alignment(tmp_path):
    config = _write_synthetic_dataset(tmp_path)
    dataset = agibot.AgibotDataset(config, action_horizon=5, load_images=False)

    assert len(dataset) == 3
    assert dataset.dataset_lengths == (3,)
    sample = dataset[0]
    assert sample["prompt"] == "pick the item"
    assert sample["state"].shape == (32,)
    assert sample["actions"].shape == (5, 32)

    # Sample zero is global frame 2, not local frame zero.
    np.testing.assert_allclose(sample["state"][20:23], [2.25, 2.5, 2.75])
    # Current state uses observed frame 2; the first target pose uses observed frame 3.
    np.testing.assert_allclose(sample["state"][:3], [20, 21, 22])
    np.testing.assert_allclose(sample["actions"][0, :3], [30, 31, 32])
    # The command remains current-step aligned (frame 2), unlike the future pose.
    assert sample["actions"][0, 3] == 102
    assert sample["actions"][0, 13] == 202
    assert np.isfinite(sample["state"]).all()
    assert np.isfinite(sample["actions"]).all()

    legacy = agibot.AgibotDataset(
        agibot.dataclasses.replace(config, extrinsic_index_mode="legacy_local"),
        action_horizon=5,
        load_images=False,
    )
    np.testing.assert_allclose(legacy[0]["state"][20:23], [0.25, 0.5, 0.75])


@pytest.mark.parametrize(("terminal", "expected_valid_steps"), [(True, 4), (False, 1)])
def test_terminal_repeat_and_padded_dimension_masks(tmp_path, terminal, expected_valid_steps):
    config = _write_synthetic_dataset(tmp_path, terminal=terminal)
    dataset = agibot.AgibotDataset(config, action_horizon=6, load_images=False)
    sample = dataset[-1]
    mask = sample["action_loss_mask"]

    assert mask.shape == (6, 32)
    np.testing.assert_array_equal(mask[:, 0], np.arange(6) < expected_valid_steps)
    assert mask[:, :20].sum() == expected_valid_steps * 20
    assert not mask[:, 20:].any()
    # All out-of-range targets repeat the last observed frame.
    np.testing.assert_allclose(sample["actions"][1:, :3], np.repeat(sample["actions"][0:1, :3], 5, axis=0))


def test_balanced_five_way_image_dropout(tmp_path, monkeypatch):
    config = _write_synthetic_dataset(tmp_path)
    dataset = agibot.AgibotDataset(config, action_horizon=2, load_images=True)
    subdataset = dataset.datasets[0]
    values = {"head_color": 1, "hand_left_color": 2, "hand_right_color": 3}
    monkeypatch.setattr(
        subdataset,
        "_read_image",
        lambda _video_path, view_name, _global_frame: np.full((2, 3, 3), values[view_name], dtype=np.uint8),
    )
    expected_kept = (
        {"head_color", "hand_left_color", "hand_right_color"},
        {"head_color"},
        {"hand_left_color", "hand_right_color"},
        {"head_color", "hand_right_color"},
        {"head_color", "hand_left_color"},
    )

    for case, kept in enumerate(expected_kept):
        monkeypatch.setattr(np.random, "randint", lambda _upper, case=case: case)
        sample = dataset[0]
        for name, value in values.items():
            assert bool(sample["image_masks"][name]) == (name in kept)
            if name in kept:
                assert np.all(sample["images"][name] == value)
            else:
                assert not sample["images"][name].any()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_pytorch_transform_and_loader_preserve_action_mask(tmp_path, num_workers):
    native_config = _write_synthetic_dataset(tmp_path)
    model_config = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=6, max_token_len=48)
    stats = {
        key: transforms.NormStats(
            # JSON-loaded stats are float64; normalization must preserve sample dtype.
            mean=np.zeros(32, dtype=np.float64),
            std=np.ones(32, dtype=np.float64),
            q01=np.zeros(32, dtype=np.float64),
            q99=np.full(32, 300, dtype=np.float64),
        )
        for key in ("state", "actions")
    }
    config = training_config.DataConfig(
        repo_id="synthetic_agibot",
        norm_stats=stats,
        data_transforms=transforms.Group(inputs=[agibot_policy.AgibotInputs()]),
        model_transforms=training_config.ModelTransformFactory()(model_config),
        use_quantile_norm=True,
        agibot_dataset=native_config,
    )
    loader = data_loader.create_torch_data_loader(
        config,
        model_config,
        action_horizon=6,
        batch_size=2,
        framework="pytorch",
        num_batches=1,
        num_workers=num_workers,
    )
    observation, actions = next(iter(loader))

    assert observation.state.shape == (2, 32)
    assert observation.state.dtype == torch.float32
    assert actions.shape == (2, 6, 32)
    assert actions.dtype == torch.float32
    assert observation.action_loss_mask.shape == actions.shape
    assert observation.action_loss_mask.dtype == torch.bool
    assert observation.tokenized_prompt.shape == (2, 48)
    assert observation.tokenized_prompt.dtype == torch.int32
    assert set(observation.images) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (2, 3, 224, 224) for image in observation.images.values())
    assert all(image.dtype == torch.float32 for image in observation.images.values())
