import torch

from openpi.models_pytorch import pi0_pytorch


def test_embed_image_with_mask_skips_invalid_images():
    calls = []
    projection = torch.nn.Linear(3, 2, bias=False)
    images = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    mask = torch.tensor([True, False, True, False])

    def embed_image(batch):
        calls.append(len(batch))
        return projection(batch).unsqueeze(1)

    result = pi0_pytorch.embed_image_with_mask(embed_image, images, mask)

    assert calls == [2]
    torch.testing.assert_close(result[mask], projection(images[mask]).unsqueeze(1))
    torch.testing.assert_close(result[~mask], torch.zeros_like(result[~mask]))
    result.sum().backward()
    torch.testing.assert_close(images.grad[~mask], torch.zeros_like(images.grad[~mask]))
    assert projection.weight.grad is not None


def test_embed_image_with_mask_all_invalid_encodes_one_shape_probe():
    calls = []
    projection = torch.nn.Linear(3, 2, bias=False)
    images = torch.ones(4, 3)

    def embed_image(batch):
        calls.append(len(batch))
        return projection(batch).unsqueeze(1)

    result = pi0_pytorch.embed_image_with_mask(embed_image, images, torch.zeros(4, dtype=torch.bool))

    assert calls == [1]
    torch.testing.assert_close(result, torch.zeros(4, 1, 2))
    result.sum().backward()
    torch.testing.assert_close(projection.weight.grad, torch.zeros_like(projection.weight.grad))
