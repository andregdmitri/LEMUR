import torch
from PIL import Image

from utils.transforms import build_train_transform


def test_build_train_transform_supports_retina_augmentation():
    transform = build_train_transform("retina_strong", 224)
    img = Image.new("RGB", (256, 256), color=(123, 45, 67))

    out = transform(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.shape[1] == 224
    assert out.shape[2] == 224


def test_build_train_transform_rejects_unknown_mode():
    try:
        build_train_transform("not_a_mode", 224)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown augmentation mode")
