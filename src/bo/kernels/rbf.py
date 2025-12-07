import torch
import structlog

LOGGER = structlog.get_logger(__name__)


def rbf_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: torch.nn.Parameter,
    variance: torch.nn.Parameter,
) -> torch.Tensor:
    """
    Kernel function:
        k(x,x') = variance * exp(- 2 * lengthscale**2 * || x - x'||**2)
    Args:
        X1 (n, d)
        X2 (m, d)
        lengthscale (1)
        variance (1)
    Returns:
        K (n, m)
    """
    norm_X1 = torch.sum(X1**2, dim=1, keepdim=True)  # (n,)
    LOGGER.debug(norm_X1.shape)

    norm_X2 = torch.sum(X2**2, dim=1).unsqueeze(0)  # (m,)
    LOGGER.debug(norm_X2.shape)

    cross_term_X1_X2 = 2 * X1 @ X2.T  # (n, m)
    LOGGER.debug(cross_term_X1_X2.shape)

    squared_distances = norm_X1 + norm_X2 - cross_term_X1_X2
    scaled_squared_distances = squared_distances / (2 * lengthscale**2)
    LOGGER.debug(scaled_squared_distances.shape)

    return variance * torch.exp(-scaled_squared_distances)  # (n, m)
