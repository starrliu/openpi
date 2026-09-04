import io
import json

import numpy as np
from PIL import Image

from openpi.serving.robotcontrol_cogact import RobotControlAdapter
from openpi.serving.robotcontrol_cogact import create_app
from openpi.training.agibot_dataset import matrix_to_cogact_rotation_6d


def _payload() -> dict:
    left_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    right_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, 3] = [0.2, -0.1, 0.7]
    return {
        "task_description": "  Pick Up Item  ",
        "state": {
            "ROBOT_LEFT_TRANS": [0.1, 0.2, 0.3],
            "ROBOT_LEFT_ROT_MAT": left_rotation.tolist(),
            "ROBOT_LEFT_GRIPPER": [0.4],
            "ROBOT_RIGHT_TRANS": [0.5, 0.6, 0.7],
            "ROBOT_RIGHT_ROT_MAT": right_rotation.tolist(),
            "ROBOT_RIGHT_GRIPPER": [0.8],
        },
        "head_camera_in_world": camera.tolist(),
    }


def _native_actions(horizon: int = 2) -> np.ndarray:
    payload = _payload()
    state = payload["state"]
    row = np.concatenate(
        (
            [1.1, 1.2, 1.3, 0.1],
            matrix_to_cogact_rotation_6d(np.asarray(state["ROBOT_LEFT_ROT_MAT"])),
            [2.1, 2.2, 2.3, 0.9],
            matrix_to_cogact_rotation_6d(np.asarray(state["ROBOT_RIGHT_ROT_MAT"])),
        )
    )
    return np.repeat(row[None], horizon, axis=0).astype(np.float32)


class _FakePolicy:
    def __init__(self):
        self.observation = None

    def infer(self, observation):
        self.observation = observation
        return {"actions": _native_actions(), "policy_timing": {"infer_ms": 1}}


def test_pack_state_uses_openpi_cogact_layout():
    payload = _payload()
    packed = RobotControlAdapter.pack_state(payload["state"], payload["head_camera_in_world"])

    assert packed.shape == (32,)
    np.testing.assert_allclose(packed[0:4], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(packed[10:14], [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(packed[20:23], [0.2, -0.1, 0.7])
    np.testing.assert_allclose(packed[23:32], np.eye(3).reshape(9))


def test_infer_builds_native_observation_and_decodes_actions():
    policy = _FakePolicy()
    images = [Image.new("RGB", (32 + index, 24), color=(index, 2, 3)) for index in range(3)]
    result = RobotControlAdapter(policy).infer(images, _payload())

    assert policy.observation["prompt"] == "pick up item"
    assert list(policy.observation["images"]) == ["head_color", "hand_left_color", "hand_right_color"]
    assert all(policy.observation["image_masks"].values())
    assert policy.observation["images"]["head_color"].shape == (24, 32, 3)
    assert set(result) == {
        "ROBOT_LEFT_TRANS",
        "ROBOT_LEFT_ROT_MAT",
        "ROBOT_LEFT_GRIPPER",
        "ROBOT_RIGHT_TRANS",
        "ROBOT_RIGHT_ROT_MAT",
        "ROBOT_RIGHT_GRIPPER",
    }
    assert len(result["ROBOT_LEFT_TRANS"]) == 2
    np.testing.assert_allclose(result["ROBOT_LEFT_ROT_MAT"][0], _payload()["state"]["ROBOT_LEFT_ROT_MAT"])
    np.testing.assert_allclose(result["ROBOT_RIGHT_ROT_MAT"][0], _payload()["state"]["ROBOT_RIGHT_ROT_MAT"])


def test_http_endpoint_accepts_robotcontrol_multipart():
    policy = _FakePolicy()
    client = create_app(policy).test_client()
    data = {"json": (io.BytesIO(json.dumps(_payload()).encode()), "data.json")}
    for index in (2, 0, 1):
        image_file = io.BytesIO()
        Image.new("RGB", (20 + index, 10), color=(index * 20, 3, 4)).save(image_file, format="JPEG")
        image_file.seek(0)
        data[f"image_{index}"] = (image_file, f"image_{index}.jpg")

    response = client.post("/api/inference", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert len(response.get_json()["ROBOT_RIGHT_TRANS"]) == 2
    assert policy.observation["images"]["hand_right_color"].shape == (10, 22, 3)


def test_http_endpoint_requires_json_part():
    response = create_app(_FakePolicy()).test_client().post("/api/inference", data={})
    assert response.status_code == 400
