import torch
from bo.kernels.rbf import RBFKernel


def test_rbf_shape():
    """Test that K has the right shape"""
    rbf_kernel = RBFKernel()
    X1 = torch.tensor([[0.0], [1.0]])
    X2 = torch.tensor([[0.0], [2.0]])
    result = rbf_kernel.forward(X1, X2)

    # K has shape (n, m)
    assert result.shape == torch.Size([X1.shape[0], X2.shape[0]])


def test_rbf_diagonal_equals_variance():
    """Test that diagonal elements equal variance**2 when X1 == X2 (distance = 0)"""
    sigma_f = torch.tensor(2.5)
    rbf_kernel = RBFKernel(log_lengthscale=0.0, log_sigma_f=torch.log(sigma_f))
    X = torch.tensor([[0.0], [1.0], [2.0]])
    K = rbf_kernel.forward(X)

    # Diagonal should equal variance (sigma_f^2)
    assert torch.allclose(K.diag(), sigma_f**2, atol=1e-6)

    # Make sure kernel is symmetric
    assert torch.allclose(K, K.T, atol=1e-6)


def test_rbf_decreases_with_distance():
    """Test that kernel value decreases as distance between points increases"""
    rbf_kernel = RBFKernel()
    X1 = torch.tensor([[0.0]])
    X2 = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    K = rbf_kernel.forward(X1, X2)

    # Kernel values should decrease as distance increases
    # K[0, 0] > K[0, 1] > K[0, 2] > K[0, 3]
    assert K[0, 0] > K[0, 1]
    assert K[0, 1] > K[0, 2]
    assert K[0, 2] > K[0, 3]
    assert K[0, 3] > 0  # Should still be positive


def test_rbf_variance_scaling():
    """Test that kernel scales linearly with variance parameter"""
    variance1 = torch.Tensor([1.0])
    rbf_kernel_1 = RBFKernel(log_sigma_f=torch.log(torch.sqrt(variance1)))

    variance2 = 2 * variance1
    rbf_kernel_2 = RBFKernel(log_sigma_f=torch.log(torch.sqrt(variance2)))

    X1 = torch.tensor([[0.0], [1.0]])
    X2 = torch.tensor([[0.0], [1.0]])
    K1 = rbf_kernel_1(X1, X2)
    K2 = rbf_kernel_2(X1, X2)

    # K2 should be exactly 2 * K1
    assert torch.allclose(K2, 2 * K1, atol=1e-6)


def test_rbf_lengthscale_effect():
    """Test that larger lengthscale makes kernel decay slower"""
    X1 = torch.tensor([[0.0]])
    X2 = torch.tensor([[2.0]])  # Distance = 2

    rbf_kernel_small_lengthscale = RBFKernel(log_lengthscale=0.5)
    rbf_kernel_large_lengthscale = RBFKernel(log_lengthscale=2.0)

    K_small = rbf_kernel_small_lengthscale(X1, X2)
    K_large = rbf_kernel_large_lengthscale(X1, X2)

    # With larger lengthscale, the kernel should be larger (decay slower)
    assert K_large > K_small
