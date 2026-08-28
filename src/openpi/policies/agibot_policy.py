"""Input/output transforms for the CogACT AGIBot baseline."""

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class AgibotInputs(transforms.DataTransformFn):
    """Map native AGIBot names to OpenPi's fixed three-camera contract."""

    def __call__(self, data: dict) -> dict:
        output = {
            "state": np.asarray(data["state"], dtype=np.float32),
        }
        if output["state"].shape[-1] != 32:
            raise ValueError(f"Expected 32D AGIBot state, got {output['state'].shape}")

        if "images" in data:
            mapping = {
                "base_0_rgb": "head_color",
                "left_wrist_0_rgb": "hand_left_color",
                "right_wrist_0_rgb": "hand_right_color",
            }
            output["image"] = {destination: data["images"][source] for destination, source in mapping.items()}
            source_masks = data.get("image_masks", {})
            output["image_mask"] = {
                destination: np.asarray(source_masks.get(source, True), dtype=bool)
                for destination, source in mapping.items()
            }

        if "actions" in data:
            output["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "action_loss_mask" in data:
            output["action_loss_mask"] = np.asarray(data["action_loss_mask"], dtype=bool)
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class AgibotOutputs(transforms.DataTransformFn):
    """Remove OpenPi's 12 padded action dimensions at inference time."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :20]}
