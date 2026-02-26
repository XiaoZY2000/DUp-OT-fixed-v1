"""
Gaussian Mixture Model fitting and PyTorch wrapper.

Supports two modes controlled by config `gmm.trainable`:
  - trainable=True:  means/log_var are nn.Parameters (fine-tuned during training)
  - trainable=False: means/log_var are registered buffers (frozen after sklearn fit)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


# ============================================================
# Sklearn GMM fitting + pruning
# ============================================================

def _compute_precisions_cholesky_diag(covariances, reg=1e-12):
    return 1.0 / np.sqrt(covariances + reg)


def _compute_precisions_cholesky_full(covariances, reg=1e-8):
    K, D, _ = covariances.shape
    precisions_chol = np.empty_like(covariances)
    for k in range(K):
        C = covariances[k] + reg * np.eye(D)
        P = np.linalg.inv(C)
        precisions_chol[k] = np.linalg.cholesky(P)
    return precisions_chol


def prune_gmm_components(gmm, weight_threshold=0.01, min_components=2,
                         reg_covar=1e-6):
    """Prune low-weight components from a fitted GMM."""
    cov_type = gmm.covariance_type
    w = gmm.weights_.copy()
    means = gmm.means_.copy()
    covs = gmm.covariances_.copy()

    idx = np.arange(len(w))
    keep = idx[w >= weight_threshold]
    if keep.size < min_components:
        keep = idx[np.argsort(-w)[:min_components]]

    w_new = w[keep]
    w_new = w_new / w_new.sum()
    means_new = means[keep]
    covs_new = covs[keep]

    K_new, D = means_new.shape
    new = GaussianMixture(
        n_components=K_new, covariance_type=cov_type,
        reg_covar=reg_covar, random_state=getattr(gmm, "random_state", None))
    new.weights_ = w_new
    new.means_ = means_new
    new.covariances_ = covs_new

    if cov_type == "diag":
        new.precisions_cholesky_ = _compute_precisions_cholesky_diag(covs_new, reg=reg_covar)
    elif cov_type == "full":
        new.precisions_cholesky_ = _compute_precisions_cholesky_full(covs_new, reg=reg_covar)
    else:
        raise NotImplementedError(f"Only diag/full supported, got {cov_type}")

    new.converged_ = True
    new.n_iter_ = getattr(gmm, "n_iter_", 0)
    new.lower_bound_ = getattr(gmm, "lower_bound_", None)
    return new


def fit_gmm_to_items(item_embeddings: torch.Tensor, cfg_gmm: dict):
    """
    Fit a Bayesian GMM to item embeddings with DP auto-pruning.

    Parameters
    ----------
    item_embeddings : torch.Tensor
        [N, D] item embeddings.
    cfg_gmm : dict
        GMM config from config.yaml.

    Returns
    -------
    Pruned sklearn GaussianMixture object.
    """
    X = item_embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
    N = X.shape[0]

    n_components = cfg_gmm.get("n_components")
    if n_components is None:
        min_k = cfg_gmm.get("prune_min_components", 2)
        k_max = int(np.sqrt(max(N, 1)))
        n_components = min(128, max(min_k, k_max))

    gmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=cfg_gmm.get("covariance_type", "diag"),
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=cfg_gmm.get("weight_concentration_prior"),
        max_iter=cfg_gmm.get("n_iters", 300),
        reg_covar=cfg_gmm.get("reg_covar", 1e-5),
        init_params="kmeans",
        tol=cfg_gmm.get("tol", 1e-3),
        random_state=cfg_gmm.get("random_state", 42),
        verbose=1,
    ).fit(X)

    gmm = prune_gmm_components(
        gmm,
        weight_threshold=cfg_gmm.get("prune_weight_threshold", 0.01),
        min_components=cfg_gmm.get("prune_min_components", 2),
        reg_covar=cfg_gmm.get("reg_covar", 1e-5),
    )
    return gmm


# ============================================================
# PyTorch GMM wrapper (trainable / frozen)
# ============================================================

class GMMWrapper(nn.Module):
    """
    PyTorch wrapper for sklearn GMM with two modes:

    - trainable=True:  means and log_var are nn.Parameters (gradient flows through)
    - trainable=False: means and log_var are registered buffers (frozen)

    Both modes expose .means_t and .prec_t properties for drop-in use
    with compute_weighted_neg_mahalanobis().
    """

    def __init__(self, sklearn_gmm, trainable: bool = True):
        super().__init__()
        self._n_components = sklearn_gmm.n_components
        self._trainable = trainable

        means = torch.tensor(sklearn_gmm.means_, dtype=torch.float32)
        cov = torch.tensor(sklearn_gmm.covariances_, dtype=torch.float32).clamp(min=1e-8)
        log_var = torch.log(cov)

        if trainable:
            self.means = nn.Parameter(means)        # [K, D]
            self.log_var = nn.Parameter(log_var)     # [K, D]
        else:
            self.register_buffer("means", means)
            self.register_buffer("log_var", log_var)

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def means_t(self) -> torch.Tensor:
        """Component means [K, D]."""
        return self.means

    @property
    def prec_t(self) -> torch.Tensor:
        """Diagonal precision = 1/variance [K, D], always > 0."""
        return torch.exp(-self.log_var)
