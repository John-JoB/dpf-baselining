
import pydpf
import torch
from torch import Tensor
from pydpf import constrained_parameter


class HestonPrior(pydpf.Module):
    def __init__(self, theta:Tensor = None, device = torch.device('cpu'), generator = torch.default_generator):
        super().__init__()
        self.device = device
        self.generator = generator
        if theta is None:
            theta = torch.rand(1, device = device, generator = generator)*1e-3
            self.log_theta = torch.nn.Parameter(torch.log(theta), requires_grad=True)
        elif isinstance(theta, torch.nn.Parameter):
            self.log_theta = torch.nn.Parameter(torch.log(theta.data), requires_grad=True)
        else:
            self.log_theta = torch.log(theta)

        self.dist = pydpf.distributions.MultivariateGaussian(torch.tensor([1], device=device, dtype=torch.float32), cholesky_covariance=torch.tensor([[1]], device = device, dtype=torch.float32)*0.5, generator=self.generator)

    def sample(self, batch_size:int, n_particles:int, **data):
        #jiggle is multiplicative in linear space, that's probably ok
        random_start =  self.log_theta + torch.log(torch.abs(self.dist.sample((batch_size, n_particles))))
        return torch.concat((random_start, torch.empty_like(random_start)), dim=-1)

    def log_density(self, state, **data):
        return self.dist.log_density(state[:,:,0:1])


class HestonDynamic(pydpf.Module):
    def __init__(self, prior:HestonPrior, k:Tensor = None, sigma = None, generator = torch.default_generator):
        super().__init__()
        self.device = prior.device
        self.generator = generator
        self.log_theta = prior.log_theta
        self.k_ = k
        if self.k_ is None:
            self.k_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator), requires_grad=True)
        self.sigma_ = sigma
        if self.sigma_ is None:
            self.sigma_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*1e-2, requires_grad=True)
        self.dist = pydpf.distributions.MultivariateGaussian(torch.tensor([0], device=self.device), cholesky_covariance=torch.tensor([[1]], device=self.device, dtype=torch.float32), generator=self.generator)

    def sample(self, prev_state, time, prev_time, **data):
        #Only allow one Euler-Maruyama step per time-step to keep the densities tractable.
        #Or at least I don't know enough SDE theory to know how to evaluate the densities after multiple E-M steps
        time_delta = pydpf.multiple_unsqueeze(time - prev_time, 2, -1)
        volatility = torch.exp(-prev_state[:,:,0:1])
        positive_drift = torch.exp(self.log_theta - prev_state[:,:,0:1])
        drift = self.k * (positive_drift- 1) - ((volatility*(self.sigma**2))/2)
        sd = self.sigma * torch.sqrt(volatility)
        random_values = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        new_state = prev_state[:,:,0:1] + drift*time_delta + sd*torch.sqrt(time_delta)*random_values
        return torch.concat((new_state, prev_state[:,:,0:1]), dim = -1)

    def log_density(self, state, prev_state, time, prev_time, **data):
        time_delta = pydpf.multiple_unsqueeze(time - prev_time, 2, -1)
        volatility = torch.exp(-prev_state[:, :, 0:1])
        positive_drift = torch.exp(self.log_theta - prev_state[:, :, 0:1])
        drift = self.k * (positive_drift - 1) - ((volatility * (self.sigma ** 2)) / 2)
        sd = self.sigma * torch.sqrt(volatility*time_delta)
        return self.dist.log_density((state[:,:,0:1] - prev_state[:,:,0:1] - drift*time_delta)/sd) - torch.log(sd.squeeze())

    @constrained_parameter
    def k(self):
        return self.k_, torch.abs(self.k_)

    @constrained_parameter
    def sigma(self):
        return self.sigma_, torch.abs(self.sigma_)

class HestonMeasurement(pydpf.Module):
    def __init__(self, dynamic:HestonDynamic, r:Tensor = None, rho:Tensor = None, generator = torch.default_generator):
        super().__init__()
        self.device = dynamic.device
        self.generator = generator
        self.r_ = r
        if self.r_ is None:
            self.r_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*1e-3, requires_grad=True)
        self.rho_ = rho
        if self.rho_ is None:
            self.rho_ = torch.nn.Parameter(torch.rand(1, device = self.device, generator = generator)*2 - 1, requires_grad=True)
        self.sigma = dynamic.sigma
        self.k = dynamic.k
        self.log_theta = dynamic.log_theta
        self.dist = pydpf.distributions.MultivariateGaussian(torch.tensor([0], device=self.device), cholesky_covariance=torch.tensor([[1]], device=self.device, dtype=torch.float32), generator=self.generator)

    @constrained_parameter
    def rho(self):
        return self.rho_, torch.clip(self.rho_, -1, 1)

    @constrained_parameter
    def r(self):
        return self.r_, torch.abs(self.r_)

    @pydpf.cached_property
    def cut_off(self):
        return 2*(torch.log(self.sigma) - torch.log(torch.tensor(2)) - 1)

    def score(self, state, time, observation, t, **data):
        if t == 0:
            #Return not defined for first time-step, so just assign all particles the same weight
            return torch.where(state[:,:,0] > -10,  0, torch.full_like(state[:,:,0], -1e8))
        prev_time = data['prev_time']
        time_delta = (time - prev_time).unsqueeze(1)
        prev_volatility = torch.exp(-state[:, :, 1])
        positive_drift = torch.exp(self.log_theta - state[:, :, 1])
        independent_mean = (self.r - (1/(2*prev_volatility)))*time_delta
        dependent_correction = (self.rho/(self.sigma*prev_volatility)) * (state[:,:,0] - state[:,:,1] - (self.k * (positive_drift - 1) - (prev_volatility*(self.sigma**2)/2))*time_delta)
        zero_mean_obs = observation - independent_mean - dependent_correction
        sd = torch.sqrt((time_delta/prev_volatility)*(1-self.rho**2))
        return torch.where(torch.logical_and(state[:,:,0] > self.cut_off, state[:,:,0] < 0),  self.dist.log_density((zero_mean_obs/sd).unsqueeze(-1)) - torch.log(sd), torch.full_like(state[:,:,0], -1e8))

    def sample(self, state, time, t, **data):
        prev_time = data['prev_time']
        time_delta = (time - prev_time).unsqueeze(1)
        prev_volatility = torch.exp(-state[:, :, 1])
        positive_drift = torch.exp(self.log_theta - state[:, :, 1])
        independent_mean = (self.r - (1 / (2 * prev_volatility))) * time_delta
        dependent_correction = (self.rho / (self.sigma * prev_volatility)) * (state[:, :, 0] - state[:, :, 1] - (self.k * (positive_drift - 1) - (prev_volatility * (self.sigma ** 2) / 2)) * time_delta)
        obs_mean = independent_mean + dependent_correction
        sd = torch.sqrt((time_delta / prev_volatility) * (1 - self.rho ** 2))
        random_values = self.dist.sample((state.size(0), state.size(1)))
        return obs_mean + sd*random_values