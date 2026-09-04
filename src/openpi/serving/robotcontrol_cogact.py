"""CogACT-compatible HTTP adapter for the OpenPi AGIBot policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import io
import json
import pathlib
import threading
from typing import Any, Protocol

from flask import Flask
from flask import jsonify
from flask import request
import numpy as np
from PIL import Image

from openpi.training.agibot_dataset import cogact_rotation_6d_to_matrix
from openpi.training.agibot_dataset import matrix_to_cogact_rotation_6d

_IMAGE_NAMES = ("head_color", "hand_left_color", "hand_right_color")


class Policy(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


class RobotControlAdapter:
    """Translate RobotControl requests to and from the native OpenPi layout."""

    def __init__(self, policy: Policy):
        self._policy = policy

    @staticmethod
    def pack_state(state: Mapping[str, Any], head_camera_in_world: Any) -> np.ndarray:
        """Pack the standard bilateral state into the 32D training layout."""

        camera = np.asarray(head_camera_in_world, dtype=np.float32)
        if camera.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 head_camera_in_world matrix, got {camera.shape}")

        packed_arms = []
        for side in ("LEFT", "RIGHT"):
            prefix = f"ROBOT_{side}"
            translation = np.asarray(state[f"{prefix}_TRANS"], dtype=np.float32)
            rotation = np.asarray(state[f"{prefix}_ROT_MAT"], dtype=np.float32)
            gripper = np.asarray(state[f"{prefix}_GRIPPER"], dtype=np.float32)
            if translation.shape != (3,) or rotation.shape != (3, 3) or gripper.size != 1:
                raise ValueError(f"Invalid {side.lower()} arm state shapes")
            packed_arms.extend((translation, gripper.reshape(1), matrix_to_cogact_rotation_6d(rotation)))

        return np.concatenate(
            (*packed_arms, camera[:3, 3], camera[:3, :3].reshape(9)),
            dtype=np.float32,
        )

    @staticmethod
    def unpack_actions(actions: Any) -> dict[str, list[Any]]:
        """Decode unnormalized OpenPi actions into RobotControl's action dict."""

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 20:
            raise ValueError(f"Expected actions shaped [horizon, >=20], got {actions.shape}")
        compact = actions[:, :20]
        return {
            "ROBOT_LEFT_TRANS": compact[:, 0:3].tolist(),
            "ROBOT_LEFT_GRIPPER": compact[:, 3:4].tolist(),
            "ROBOT_LEFT_ROT_MAT": cogact_rotation_6d_to_matrix(compact[:, 4:10]).tolist(),
            "ROBOT_RIGHT_TRANS": compact[:, 10:13].tolist(),
            "ROBOT_RIGHT_GRIPPER": compact[:, 13:14].tolist(),
            "ROBOT_RIGHT_ROT_MAT": cogact_rotation_6d_to_matrix(compact[:, 14:20]).tolist(),
        }

    def infer(self, images: Sequence[Image.Image], payload: Mapping[str, Any]) -> dict[str, list[Any]]:
        if len(images) != len(_IMAGE_NAMES):
            raise ValueError(f"Expected three RGB images, got {len(images)}")
        observation = {
            "state": self.pack_state(payload["state"], payload["head_camera_in_world"]),
            "images": {
                name: np.asarray(image.convert("RGB"), dtype=np.uint8)
                for name, image in zip(_IMAGE_NAMES, images, strict=True)
            },
            "image_masks": dict.fromkeys(_IMAGE_NAMES, True),
            "prompt": str(payload.get("task_description", "")).strip().lower(),
        }
        return self.unpack_actions(self._policy.infer(observation)["actions"])


def create_app(policy: Policy) -> Flask:
    """Create the small Flask service without loading a checkpoint."""

    app = Flask(__name__)
    adapter = RobotControlAdapter(policy)
    inference_lock = threading.Lock()

    @app.post("/api/inference")
    def inference():
        indexed_images = []
        try:
            for key, image_file in request.files.items():
                if key.startswith("image_"):
                    indexed_images.append((int(key[6:]), Image.open(io.BytesIO(image_file.read())).convert("RGB")))
            indexed_images.sort(key=lambda item: item[0])
            query_file = request.files.get("json")
            if query_file is None:
                return jsonify({"error": "Missing json file"}), 400
            payload = json.load(query_file)
            with inference_lock:
                answer = adapter.infer([image for _, image in indexed_images], payload)
            return jsonify(answer)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, Image.UnidentifiedImageError) as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/reset")
    def reset():
        return jsonify({"status": "reset"})

    return app


def load_policy(
    checkpoint_dir: str,
    *,
    config_name: str = "pi05_cogact_baseline",
    pytorch_device: str | None = None,
) -> Policy:
    """Validate and load the trained PI0.5 policy, resolving its manifest from the repo root."""

    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as training_config

    checkpoint = pathlib.Path(download.maybe_download(checkpoint_dir))
    config = training_config.get_config(config_name)
    if not checkpoint.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    asset_id = config.data.repo_id
    if not checkpoint.joinpath("assets", asset_id, "norm_stats.json").is_file():
        raise FileNotFoundError(checkpoint / "assets" / asset_id / "norm_stats.json")

    manifest = getattr(config.data, "dataset_manifest", None)
    if manifest is not None and not pathlib.Path(manifest).is_absolute():
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, dataset_manifest=str(repo_root / manifest))
        )
    return policy_config.create_trained_policy(config, checkpoint, pytorch_device=pytorch_device)
