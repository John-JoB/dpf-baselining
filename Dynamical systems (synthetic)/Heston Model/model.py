import pydpf
import torch
from typing import Union
from torch import Tensor

from pydpf import constrained_parameter


class HestonPrior(pydpf.Module):
    def __init__(self, theta:Tensor = None, device = torch.device('cpu'), generator = torch.default_generator):
        super().__init__()
        self.device = device
        self.theta_ = theta
        self.generator = generator
        if self.theta_ is None:
            self.theta_ = torch.nn.Parameter(torch.rand(1, device = device, generator = generator)*1e-3, requires_grad=True)
        self.dist = pydpf.distributions.MultivariateGaussian(self.theta, cholesky_covariance=torch.tensor([[1]], device = device), generator=self.generator)

    @constrained_parameter
    def theta(self):
        return self.theta_, torch.abs(self.theta_)

    def sample(self, batch_size:int, n_particles:int, **data):
        #jiggle is multiplicative in linear space, that's probably ok
        random_jiggle =  self.dist.sample((batch_size, n_particles))
        return torch.concat((random_jiggle + self.theta, torch.empty_like(random_jiggle)), dim=-1)

    def log_density(self, state, **data):
        return self.dist.log_density(state[:,:,0:1])


class HestonDynamic(pydpf.Module):
    def __init__(self, prior:HestonPrior, k:Tensor = None, sigma = None, generator = torch.default_generator):
        super().__init__()
        self.device = prior.device
        self.generator = generator
        self.theta = prior.theta
        self._k = k
        if self.k is None:
            self._k = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator), requires_grad=True)
        self.sigma_ = sigma
        if self.sigma_ is None:
            self.sigma_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*1e-2, requires_grad=True)
        self.dist = pydpf.distributions.MultivariateGaussian(torch.tensor([1], device=self.device), cholesky_covariance=torch.tensor([[1]], device=self.device), generator=self.generator)

    def sample(self, prev_state, time, prev_time, **data):
        #Only allow one Euler-Maruyama step per time-step to keep the densities tractable.
        #Or at least I don't know enough SDE theory to know how to evaluate the densities after multiple E-M steps
        time_delta = time - prev_time
        volatility = torch.exp(-prev_state[:,:,0:1])
        drift = self.k * (self.theta*volatility - 1) - ((volatility*(self.sigma**2))/2)
        sd = self.sigma * torch.sqrt(volatility)
        random_values = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        new_state = prev_state[:,:,0:1] + drift*time_delta + sd*torch.sqrt(time_delta)*random_values
        return torch.concat((new_state, prev_state[:,:,0:1]), dim = -1)

    def log_density(self, state, prev_state, time, prev_time, **data):
        time_delta = time - prev_time
        volatility = torch.exp(-prev_state[:, :, 0:1])
        drift = self.k * (self.theta * volatility - 1) - ((volatility * (self.sigma ** 2)) / 2)
        sd = self.sigma * torch.sqrt(volatility)
        return self.dist.log_density((state[:,:,0:1] - prev_state[:,:,0:1] - drift*time_delta)/(sd*torch.sqrt(time_delta)))

    @constrained_parameter
    def k(self):
        return self._k, torch.abs(self._k)

    @constrained_parameter
    def sigma(self):
        return self.sigma_, torch.abs(self.sigma_)

class HestonMeasurement(pydpf.Module):
    def __init__(self, dynamic:HestonDynamic, r:Tensor = None, rho:Tensor = None, generator = torch.default_generator):
        super().__init__()
        self.device = dynamic.device
        self.r_ = r
        if self.r_ is None:
            self.r_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*1e-3, requires_grad=True)
        self.rho_ = rho
        if self.rho_ is None:
            self.rho_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*2 - 1, requires_grad=True)

    @constrained_parameter
    def rho(self):
        return self.rho_, torch.clip(self.rho_, -1, 1)

    @constrained_parameter
    def r(self):
        return self.r_, torch.abs(self.r_)