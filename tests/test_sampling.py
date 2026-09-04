import pytest
import torch

from h3_icr.sampling import validate_partial_sigmas


def test_partial_sigma_contract():
    assert validate_partial_sigmas(torch.tensor([0.6, 0.3, 0.0])) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="full-noise"):
        validate_partial_sigmas(torch.tensor([1.0, 0.0]))
