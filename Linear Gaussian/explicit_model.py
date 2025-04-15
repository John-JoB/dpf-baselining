'''
Shows how to create the same model as in the Experiments notebook without using the LinearGaussian class
'''
import torch
from torch import Tensor
import pydpf

def transform_Gaussian_sample(standard_sample:Tensor, mean:Tensor, cholesky_covariance:Tensor)->Tensor:
    return mean + standard_sample @ cholesky_covariance.T

class GaussianPrior(pydpf.Module):
    def __init__(self, mean:Tensor, cholesky_covariance:Tensor, device:torch.device, generator:torch.Generator):
        super().__init__()
        self.mean = mean
        self.cholesky_covariance_ = cholesky_covariance
        self.device = device
        self.generator = generator

    def sample(self, batch_size:int, n_particles:int)->Tensor:
        standard_sample = torch.randn((batch_size, n_particles, self.mean.size(0)), device=self.device, generator=self.generator)
        return self.mean + standard_sample @ self.cholesky_covariance.T

    # Constrain the cholesky_covariance to be lower-triangular with positive diagonal
    @pydpf.constrained_parameter
    def cholesky_covariance(self):
        tril = torch.tril(self.cholesky_covariance_)
        diag = tril.diagonal()
        diag.mul_(diag.sign())
        return self.cholesky_covariance_, tril


class LinearGaussianDynamic(pydpf.Module):
    def __init__(self, weight: Tensor, bias:Tensor, cholesky_covariance: Tensor, device: torch.device, generator: torch.Generator, max_spectral_radius:float = 0.99):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.cholesky_covariance_ = cholesky_covariance
        self.device = device
        self.generator = generator
        self.max_spectral_radius = max_spectral_radius

    def sample(self, prev_state:Tensor)->Tensor:
        standard_sample = torch.randn(prev_state.size(), device=self.device, generator=self.generator)
        mean = (self.constrained_weight @ prev_state.unsqueeze(-1)).squeeze() + self.bias
        return self.mean + standard_sample @ self.cholesky_covariance.T

    #Constrain the cholesky_covariance to be lower-triangular with positive diagonal
    @pydpf.constrained_parameter
    def cholesky_covariance(self):
        tril = torch.tril(self.cholesky_covariance_)
        diag = tril.diagonal()
        diag.mul_(diag.sign())
        return self.cholesky_covariance_, tril

    #Constrain the weight's spectral radius to avoid divergence
    @pydpf.constrained_parameter
    def constrained_weight(self):
        if self.max_spectral_radius is not None:
            eigvals = torch.linalg.eigvals(self.weight)
            spectral_radius = torch.max(torch.abs(eigvals))
            if spectral_radius > self.max_spectral_radius:
                return self.weight, self.weight / spectral_radius
        return self.weight, self.weight


class LinearGaussianObservation(pydpf.Module):
    def __init__(self, weight: Tensor, bias:Tensor, cholesky_covariance: Tensor, device: torch.device, generator: torch.Generator):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.cholesky_covariance_ = cholesky_covariance
        self.device = device
        self.generator = generator

    def score(self, state:Tensor, observation:Tensor)->Tensor:
        mean = (self.weight @ state.unsqueeze(-1)).squeeze() + self.bias
        residuals = observation.unsqueeze(1) - mean
        exponent = (-1 / 2) * torch.sum((residuals @ self.inv_cholesky_covariance.T) ** 2, dim=-1)
        return self.density_pre_factor + exponent

    # Constrain the cholesky_covariance to be lower-triangular with positive diagonal
    @pydpf.constrained_parameter
    def cholesky_covariance(self):
        tril = torch.tril(self.cholesky_covariance_)
        diag = tril.diagonal()
        diag.mul_(diag.sign())
        return self.cholesky_covariance_, tril

    #Cache the inverse covariance to avoid recalculating it every time it's required
    @pydpf.cached_property
    def inv_cholesky_covariance(self):
        return torch.linalg.inv_ex(self.cholesky_covariance)[0]

    #Cache the normalising constant for the density
    @pydpf.cached_property
    def density_pre_factor(self):
        return -1/2 * self.weight.size(-1) * torch.log(torch.tensor(2*torch.pi)) - torch.linalg.slogdet(self.cholesky_covariance)[1]
