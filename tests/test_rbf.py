import torch
from bo.kernels.rbf import rbf_kernel


def test_rbf():
    X1 = torch.tensor([[0.0], [1.0]])
    X2 = torch.tensor([[0.0], [2.0]])
    lengthscale = 1.0
    variance = 1.0

    # Get dimensions
    n = X1.shape[0]
    m = X2.shape[0]

    result = rbf_kernel(X1, X2, lengthscale, variance)
    assert result.shape == torch.Size([n, m])
