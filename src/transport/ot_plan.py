"""Optimal Transport: cost matrix, EMD plan, and MMD alignment loss."""

import torch
from ot import emd


def compute_cost_matrix(gmm1, gmm2, device=None, return_squared=False, eps=1e-12):
    """
    W2 (or W2²) cost matrix between two GMMs with diagonal covariances.

    W2² = ||μ₁−μ₂||² + Σ_d( v₁ + v₂ − 2√(v₁v₂) )
    """
    assert gmm1.covariance_type == 'diag' and gmm2.covariance_type == 'diag'
    device = device or torch.device("cpu")

    mu1 = torch.tensor(gmm1.means_, dtype=torch.float32, device=device)
    mu2 = torch.tensor(gmm2.means_, dtype=torch.float32, device=device)
    v1 = torch.tensor(gmm1.covariances_, dtype=torch.float32, device=device).clamp_min(eps)
    v2 = torch.tensor(gmm2.covariances_, dtype=torch.float32, device=device).clamp_min(eps)

    # Mean term
    mu1_sq = (mu1 ** 2).sum(1).unsqueeze(1)
    mu2_sq = (mu2 ** 2).sum(1).unsqueeze(0)
    mean_term = mu1_sq + mu2_sq - 2.0 * (mu1 @ mu2.T)

    # Covariance term
    s1 = v1.sum(1).unsqueeze(1)
    s2 = v2.sum(1).unsqueeze(0)
    cross = torch.sqrt(v1) @ torch.sqrt(v2).T
    cov_term = s1 + s2 - 2.0 * cross

    w2_sq = mean_term + cov_term
    if return_squared:
        return w2_sq
    return torch.sqrt(w2_sq.clamp_min(0.0))


def compute_transport_plan(cost_matrix):
    """Solve for optimal transport plan using Earth Mover's Distance."""
    n_src, n_tgt = cost_matrix.shape
    a = torch.ones(n_src) / n_src
    b = torch.ones(n_tgt) / n_tgt
    T = emd(a.numpy(), b.numpy(), cost_matrix.cpu().numpy())
    return torch.tensor(T, dtype=torch.float32)


def compute_mmd(x, y, bandwidth=None):
    """
    MMD² (Maximum Mean Discrepancy) with Gaussian RBF kernel.

    No sample-level correspondence needed — just two independent batches.

    Args:
        x: [N, D] transported source weights.
        y: [M, D] target weights (typically detached).
        bandwidth: RBF bandwidth; None uses median heuristic.
    Returns:
        Scalar MMD².
    """
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    if bandwidth is None:
        all_dists = torch.cat([xx.reshape(-1), yy.reshape(-1), xy.reshape(-1)])
        bandwidth = torch.median(all_dists).clamp(min=1e-5)
    k_xx = torch.exp(-xx / (2 * bandwidth))
    k_yy = torch.exp(-yy / (2 * bandwidth))
    k_xy = torch.exp(-xy / (2 * bandwidth))
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
