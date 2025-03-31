import pydpf
import torch
from typing import Union
from torch import Tensor

class LorenzMeasurement(pydpf.Module):
    def __init__(self, state_dim:int, obs_dim:int,  device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        root_measurement_cov = torch.nn.Parameter(torch.ones(size=(obs_dim, obs_dim), device=device) * torch.rand(size=(obs_dim,obs_dim), device = device, generator=generator))
        observation_matrix = torch.nn.Parameter(torch.ones(size=(obs_dim, state_dim), device=device) * torch.rand(size=(obs_dim,state_dim), device = device, generator=generator))
        self.observation_dist = pydpf.distributions.LinearGaussian(observation_matrix, torch.zeros(size=(obs_dim,), device=device), root_measurement_cov, diagonal_cov=False, generator=generator)

    def score(self, state:Tensor, observation:Tensor, time, prev_time = None) -> Tensor:
       return self.observation_dist.log_density(observation.unsqueeze(1), state)


class LorenzDynamic(pydpf.Module):
    def __init__(self, state_dim:int, sub_steps:int = 1, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator, forcing = None, diffusion_cov = None):
        super().__init__()
        self.sub_steps = sub_steps
        if diffusion_cov is None:
            self.diffusion_cov = torch.nn.Parameter(torch.eye(state_dim, device = device) * torch.rand(size= (state_dim  ,1), device = device, generator=generator))
        else:
            self.diffusion_cov = torch.sqrt(diffusion_cov)
        if forcing is None:
            self._forcing = torch.nn.Parameter(torch.rand(size=(1,), device=device, generator=generator).squeeze() * 10)
        else:
            self._forcing = torch.tensor(forcing, device = device)
        self.diffusion_dist = pydpf.distributions.MultivariateGaussian(mean=torch.zeros((state_dim,), device = device), cholesky_covariance=self.diffusion_cov, diagonal_cov=True, generator=generator)

    @pydpf.constrained_parameter
    def forcing(self):
        return self._forcing, torch.abs(self._forcing)

    def Lorenz_derivative(self, state:Tensor, forcing:Tensor)->Tensor:
        return (torch.roll(state, -1, -1) - torch.roll(state, 2, -1)) * torch.roll(state, 1, -1) - state + forcing

    def runge_kutta_update(self, state:Tensor, time_gap:torch.Tensor, sub_steps:int):
        state_now = state
        step_size = time_gap / sub_steps
        for i in range(sub_steps):
            k1 = self.Lorenz_derivative(state_now, self.forcing)
            k2 = self.Lorenz_derivative(state_now + step_size*k1/2, self.forcing)
            k3 = self.Lorenz_derivative(state_now + step_size*k2/2, self.forcing)
            k4 = self.Lorenz_derivative(state_now + step_size*k3, self.forcing)
            state_now = state_now + step_size*(k1 + 2*k2 + 2*k3 + k4)/6
        return state_now + self.diffusion_dist.sample((state_now.shape[0], state_now.shape[1]))

    def euler_update(self, state:Tensor, time_gap:torch.Tensor, sub_steps:int):
        state_now = state
        step_size = time_gap / sub_steps
        root_step_size = torch.sqrt(step_size)
        for i in range(sub_steps):
            state_now = state_now + self.Lorenz_derivative(state_now, self.forcing) * step_size + root_step_size * self.diffusion_dist.sample((state_now.shape[0], state_now.shape[1]))
        return state_now

    def sample(self, prev_state:Tensor, time, prev_time):
        return self.euler_update(prev_state, (time - prev_time).unsqueeze(1).unsqueeze(2), self.sub_steps)

    def log_density(self, state:Tensor, prev_state:Tensor, time, prev_time):
        if not self.sub_steps == 1:
            raise ValueError('Can only estimate the density if a single Euler step is taken per time-step')
        drift = self.Lorenz_derivative(prev_state, self.forcing)
        time_gap = (time - prev_time).unsqueeze(1).unsqueeze(2)
        root_gap = torch.sqrt(time_gap)
        state_0_mean = (state - (drift * time_gap))/root_gap
        return self.diffusion_dist.log_density(state_0_mean) - torch.log(root_gap).squeeze(-1)

class LorenzPrior(pydpf.Module):
    def __init__(self, state_dim: int, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.gen = generator
        self.initial_range = torch.nn.Parameter(torch.rand(size=(1,), device = device, generator = generator).squeeze()*0.1)

    def sample(self, batch_size:int, n_particles:int, time):
        initial_state = torch.ones(size=(batch_size, n_particles, self.state_dim), device = self.device)
        initial_state[:, :, 0] = torch.rand(size=(batch_size, n_particles), device = self.device, generator=self.gen) * self.initial_range
        return initial_state