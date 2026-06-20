import numpy as np
import torch

from PIL import Image

from src.data.openvla_action_utils import (
    OPENVLA_EMPTY_ACTION_TOKEN_ID,
    append_openvla_empty_action_token,
    apply_center_crop,
    get_libero_dummy_action,
    get_suite_max_steps,
    invert_gripper_action,
    normalize_gripper_action,
    preprocess_libero_image,
)


def test_append_openvla_empty_action_token_appends_suffix_and_mask() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    out = append_openvla_empty_action_token(inputs)

    assert out["input_ids"].tolist() == [[1, 2, 3, OPENVLA_EMPTY_ACTION_TOKEN_ID]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert inputs["input_ids"].tolist() == [[1, 2, 3]]


def test_append_openvla_empty_action_token_is_noop_when_suffix_exists() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, OPENVLA_EMPTY_ACTION_TOKEN_ID]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    out = append_openvla_empty_action_token(inputs)

    assert out["input_ids"].tolist() == inputs["input_ids"].tolist()
    assert out["attention_mask"].tolist() == inputs["attention_mask"].tolist()


def test_append_openvla_empty_action_token_leaves_non_tensor_inputs_alone() -> None:
    inputs = {"pixel_values": "unchanged"}

    out = append_openvla_empty_action_token(inputs)

    assert out == inputs


def test_apply_center_crop_preserves_size() -> None:
    image = Image.new("RGB", (224, 224), color=(10, 20, 30))

    out = apply_center_crop(image, crop_fraction=0.9)

    assert out.size == (224, 224)


def test_apply_center_crop_is_noop_for_full_fraction() -> None:
    image = Image.new("RGB", (100, 80), color=(10, 20, 30))

    out = apply_center_crop(image, crop_fraction=1.0)

    assert out.size == image.size


def test_preprocess_libero_image_rotates_and_resizes() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]
    image[1, 1] = [0, 255, 0]

    out = preprocess_libero_image(image, resize_size=2)

    assert out.size == (2, 2)
    out_np = np.asarray(out)
    assert out_np[0, 0].sum() > 0


def test_gripper_postprocess_matches_official_eval() -> None:
    action = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 1.0], dtype=np.float32)

    out = invert_gripper_action(normalize_gripper_action(action, binarize=True))

    assert np.allclose(out[:-1], action[:-1])
    assert float(out[-1]) == -1.0


def test_get_libero_dummy_action_and_suite_steps() -> None:
    dummy = get_libero_dummy_action()

    assert dummy.shape == (7,)
    assert float(dummy[-1]) == -1.0
    assert get_suite_max_steps("long") == 520
