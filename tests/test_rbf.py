import torch
from bo.kernels.rbf import rbf_kernel


def test_rbf():
    X1 = torch.tensor([[0.0], [1.0]])
    X2 = torch.tensor([[0.0], [2.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))
    variance = torch.nn.Parameter(torch.Tensor([1.0]))
    result = rbf_kernel(X1, X2, lengthscale, variance)

    # K has shape (n, m)
    assert result.shape == torch.Size([X1.shape[0], X2.shape[0]])


def test_rbf_diagonal_equals_variance():
    """Test that diagonal elements equal variance when X1 == X2 (distance = 0)"""
    X = torch.tensor([[0.0], [1.0], [2.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))
    variance = torch.nn.Parameter(torch.Tensor([2.5]))
    K = rbf_kernel(X, X, lengthscale, variance)

    # Diagonal should equal variance (exp(0) = 1)
    assert torch.allclose(K.diag(), variance, atol=1e-6)


def test_rbf_symmetry():
    """Test that kernel matrix is symmetric when X1 == X2"""
    X = torch.tensor([[0.0], [1.0], [2.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))
    variance = torch.nn.Parameter(torch.Tensor([1.0]))
    K = rbf_kernel(X, X, lengthscale, variance)

    assert torch.allclose(K, K.T, atol=1e-6)


def test_rbf_decreases_with_distance():
    """Test that kernel value decreases as distance between points increases"""
    X1 = torch.tensor([[0.0]])
    X2 = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))
    variance = torch.nn.Parameter(torch.Tensor([1.0]))
    K = rbf_kernel(X1, X2, lengthscale, variance)

    # Kernel values should decrease as distance increases
    # K[0, 0] > K[0, 1] > K[0, 2] > K[0, 3]
    assert K[0, 0] > K[0, 1]
    assert K[0, 1] > K[0, 2]
    assert K[0, 2] > K[0, 3]
    assert K[0, 3] > 0  # Should still be positive


def test_rbf_zero_distance():
    """Test that identical points give kernel value equal to variance"""
    X1 = torch.tensor([[1.0, 2.0]])
    X2 = torch.tensor([[1.0, 2.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))
    variance = torch.nn.Parameter(torch.Tensor([3.0]))
    K = rbf_kernel(X1, X2, lengthscale, variance)

    # When distance is 0, exp(0) = 1, so K = variance
    assert torch.allclose(K, variance, atol=1e-6)


def test_rbf_variance_scaling():
    """Test that kernel scales linearly with variance parameter"""
    X1 = torch.tensor([[0.0], [1.0]])
    X2 = torch.tensor([[0.0], [1.0]])
    lengthscale = torch.nn.Parameter(torch.Tensor([1.0]))

    variance1 = torch.nn.Parameter(torch.Tensor([1.0]))
    variance2 = torch.nn.Parameter(torch.Tensor([2.0]))

    K1 = rbf_kernel(X1, X2, lengthscale, variance1)
    K2 = rbf_kernel(X1, X2, lengthscale, variance2)

    # K2 should be exactly 2 * K1
    assert torch.allclose(K2, 2 * K1, atol=1e-6)


def test_rbf_lengthscale_effect():
    """Test that larger lengthscale makes kernel decay slower"""
    X1 = torch.tensor([[0.0]])
    X2 = torch.tensor([[2.0]])  # Distance = 2
    variance = torch.nn.Parameter(torch.Tensor([1.0]))

    lengthscale_small = torch.nn.Parameter(torch.Tensor([0.5]))
    lengthscale_large = torch.nn.Parameter(torch.Tensor([2.0]))

    K_small = rbf_kernel(X1, X2, lengthscale_small, variance)
    K_large = rbf_kernel(X1, X2, lengthscale_large, variance)

    # With larger lengthscale, the kernel should be larger (decay slower)
    assert K_large > K_small
