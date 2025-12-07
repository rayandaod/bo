import torch

from bo.kernels.rbf import RBFKernel


class GPRegression(torch.nn.Module):
    def __init__(self, kernel: torch.nn.Module, log_noise_variance: float):
        super().__init__()
        self.kernel = kernel
        # Observation noise
        self.log_noise_variance = torch.nn.Parameter(log_noise_variance)

    def noisy_covariance_matrix(self, X: torch.Tensor, eps: float = 1e-6):
        K = self.kernel(X)
        N = K.shape[0]
        I = torch.eye(N, device=K.device, dtype=K.dtype)  # noqa: E741
        noise = torch.exp(self.log_noise_variance)
        return K + (noise + eps) * I

    def predict():
        pass


if __name__ == "__main__":
    X = torch.rand((10, 3))
    gp_regression = GPRegression(kernel=RBFKernel(), log_noise_variance=-2.0)
    K = gp_regression.noisy_covariance_matrix(X)
    print(K.shape)
