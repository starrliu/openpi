import pytest
import torch

from openpi.training import losses


def test_masked_mean_without_mask_matches_torch_mean():
    values = torch.tensor([[1.0, 2.0], [4.0, 9.0]])
    assert torch.equal(losses.masked_mean(values), values.mean())


def test_masked_mean_ignores_masked_errors():
    values = torch.tensor([1.0, 3.0, 1e20])
    mask = torch.tensor([True, True, False])
    assert losses.masked_mean(values, mask).item() == 2.0


def test_masked_mean_empty_mask_is_differentiable_zero():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    result = losses.masked_mean(values, torch.zeros(2, dtype=torch.bool))
    assert result.item() == 0.0
    assert torch.isfinite(result)
    result.backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_masked_mean_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="mask shape"):
        losses.masked_mean(torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool))
