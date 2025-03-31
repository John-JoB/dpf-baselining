import pydpf
import torch
from typing import Union, Iterable



def simulate_ahead(horizon:int, SSM:pydpf.FilteringModel, n_particles:int, EM_steps_per_day:int = 1, target_horizons:Union[int, Iterable[int]] = None):
    if target_horizons is None:
        target_horizons = horizon
    theta = SSM.prior_model.theta
    time = (torch.arange(horizon * EM_steps_per_day)/EM_steps_per_day).unsqueeze(1)
    _, obs = pydpf.simulate(pydpf.multiple_unsqueeze(theta, 2, -1).expand(1, n_particles), 0, SSM, horizon, time=time)
    log_ratios = torch.cumsum(obs, dim=0)
    if isinstance(target_horizons, int):
        return log_ratios[target_horizons]
    horizon_dict = {}
    for h in target_horizons:
        horizon_dict[h] = log_ratios[h]
    return horizon_dict

def compare_quartiles(observation_tensor, horizon, predicted_ratios):
    empirical_ratios = observation_tensor[horizon:,0] / observation_tensor[:-horizon, 0]
    q = torch.arange(20)[1:]/20
    empirical_quartiles = torch.quantile(empirical_ratios, q)
    predicted_quartiles = torch.quantile(predicted_ratios, q)
    print(empirical_quartiles)
    print(predicted_quartiles)



