import torch


class RBFKernel(torch.nn.Module):
    def __init__(self, log_lengthscale=0.0, log_sigma_f=0.0):
        super().__init__()
        # Lengthscale encodes how quickly the function changes in input space
        self.log_lengthscale = torch.nn.Parameter(torch.tensor(log_lengthscale))
        # Controls the signal variance (overall vertical scale) of the underlying latent function f(x)
        self.log_sigma_f = torch.nn.Parameter(torch.tensor(log_sigma_f))

    def forward(self, X: torch.Tensor, X2: torch.Tensor = None):
        if X2 is None:
            X2 = X

        # We optimize the log version of these to ensure positivity,
        # so we need to come back to non-log domain first
        lengthscale = torch.exp(self.log_lengthscale)
        sigma_f2 = torch.exp(2 * self.log_sigma_f)

        norm_X = torch.sum(X**2, dim=1, keepdim=True)  # (n,)
        norm_X2 = torch.sum(X2**2, dim=1).unsqueeze(0)  # (m,)

        cross_term_X_X2 = 2 * X @ X2.T  # (n, m)
        squared_distances = norm_X + norm_X2 - cross_term_X_X2

        scaled_squared_distances = squared_distances / (2 * lengthscale**2)
        return sigma_f2 * torch.exp(-scaled_squared_distances)  # (n, m)
