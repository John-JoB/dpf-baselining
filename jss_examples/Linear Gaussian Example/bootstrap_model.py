import pydpf
import torch


def make_components(dx:int, dy:int, device):
    dynamic_matrix = 0.42 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
    dynamic_offset = torch.zeros(dx, device=device)
    dynamic_model = pydpf.LinearGaussian(weight=dynamic_matrix, bias=dynamic_offset, cholesky_covariance=torch.eye(dx, device=device), generator=torch.Generator(device=device).manual_seed(0))
    observation_matrix = torch.zeros((dy, dx), device=device)
    for i in range(dy):
        observation_matrix[i, i] = 1
    observation_offset = torch.zeros(dy, device=device)
    observation_model = pydpf.LinearGaussian(weight=observation_matrix, bias=observation_offset, cholesky_covariance=torch.eye(dy, device=device), generator=torch.Generator(device=device).manual_seed(10))
    prior_model = pydpf.MultivariateGaussian(torch.zeros(dx, device=device), torch.eye(dx, device=device), generator=torch.Generator(device=device).manual_seed(20))
    return prior_model, dynamic_model, observation_model



def make_SSM(dx:int, dy:int, device):
    prior_model, dynamic_model, observation_model = make_components(dx, dy, device)
    return pydpf.FilteringModel(prior_model = prior_model, dynamic_model = dynamic_model, observation_model = observation_model)