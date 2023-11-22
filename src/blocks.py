from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_components,
        hidden_dim,
        noise_type=NoiseType.DIAGONAL,
        fixed_noise_level=None,
    ):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_out * n_components + num_sigma_channels),
        )

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., : self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components :]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(*mu.shape[:-1], self.n_components, self.dim_out)
        sigma = sigma.reshape(*sigma.shape[:-1], self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(-2) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.prod(z_score**2, dim=-1)  # correct for multi-dimensional?
            # -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x, samples=1, squeeze=True):

        batch_dims = x.dim() - 1 # number of initial dimensions to be treated as batch

        log_pi, mu, sigma = self.forward(x) 
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)

        # expand on sample dimension
        cum_pi = cum_pi.unsqueeze(batch_dims).repeat(*[1]*batch_dims, samples, 1) # ..., S, C
        mu = mu.unsqueeze(batch_dims).repeat(*[1]*batch_dims, samples, 1, 1) # ..., S, C, O
        sigma = sigma.unsqueeze(batch_dims).repeat(*[1]*batch_dims, samples, 1, 1) # ..., S, C, O

        # sample active mode
        rvs = torch.rand(*x.shape[:-1], samples, 1)
        rand_pi = torch.searchsorted(cum_pi, rvs)

        # sample individual gaussians        
        rand_normal = torch.randn_like(mu) * sigma + mu  #..., S, C, O 

        # choose a random gaussian
        out_samples = torch.take_along_dim(
            rand_normal, indices=rand_pi.unsqueeze(-1), dim=batch_dims+1,
        ).squeeze(batch_dims+1)

        # squeeze if samples=1 and squeeze=1
        if squeeze:
            out_samples = out_samples.squeeze(batch_dims)
        return out_samples
