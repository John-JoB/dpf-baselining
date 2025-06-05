import torch
from torch import Tensor
from typing import Callable, Tuple
from .base import Module
from .distributions import KernelMixture
from .custom_types import ImportanceKernel

class FilteringMean(Module):
    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Get an estimate of the filtering mean of a function of the latent state.

        Parameters
        ----------
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.function = function

    def forward(self, *, state: Tensor, weight: Tensor, **data) -> Tensor:
        return torch.einsum('ij..., ij -> i... ', self.function(state), torch.exp(weight))

class MSE_Loss(Module):

    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Get the per-timestep mean squared error of a function of the latent state compared to ground truth over a batch of filters.

        Parameters
        ----------
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.mean = FilteringMean(function)

    def forward(self, *, state: Tensor, weight: Tensor, ground_truth, **data):
        filter_mean = self.mean(state = state, weight = weight)
        return torch.mean(torch.sum((ground_truth - filter_mean) ** 2, dim=-1))

class LogLikelihoodFactors(Module):

    def __init__(self):
        """
        Get the log observation likelihood factor for each time step.
        Such that an estimate of the log likelihood over the trajectory may be given by a sum of these factors.
        """
        super().__init__()

    def forward(self, likelihood, **kwargs) -> Tensor:
        return likelihood

class ElBO_Loss(Module):

    def __init__(self):
        """
        Get the factors of the ELBO loss per-timestep for a batch of filters.
        The complete ELBO loss may be given by a sum of these factors.

        Notes
        -----
        As an average of the log-likelihood, the ELBO estimates a Jensens' inequality lower bound to the mean log of the likelihood.
        The SMC ELBO is not exactly analogous to the variation auto-encoder ELBO, see C. Naesseth, S. Linderman, R. Ranganath, and D. Blei, 'Variational Sequential Monte-Carlo', 2018 for more information.
        """
        super().__init__()

    def forward(self, likelihood, **kwargs):
        return -torch.mean(likelihood)


class PredictiveMean(Module):

    def __init__(self, prediction_kernel: ImportanceKernel, lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Predict the state n steps ahead.

        Parameters
        ----------
        prediction_kernel: ImportanceKernel
            A function to importance sample from the predictive distribution n-steps ahead. Typically, this will entail be applying the bootstrap proposal n-times.
        lag: int
            How many steps ahead the prediction is being made.
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.prediction_kernel = prediction_kernel
        self.lag = lag
        self.function = function

    def forward(self, *, state: Tensor, norm_weights: Tensor, data, time, **kwargs):
        prediction, new_weights = self.prediction_kernel(state, norm_weights, data[time:time+self.lag].squeeze(), time)
        return torch.einsum('ij...,ij...->i...', self.function(prediction), torch.exp(new_weights))


class NegLogDataLikelihood_Loss(Module):

    def __init__(self, kernel: KernelMixture):
        """
        Get the negative log data likelihood per-timestep for a batch of kernel filters.
        This function applies a kernel density estimator over the particles and calculates the log likelihood of the ground truth given the KDE.

        Parameters
        ----------
        kernel: KernelMixture
            The kernel density estimator.
        """
        super().__init__()
        self.KDE = kernel


    def forward(self, *, state: Tensor, weight: Tensor, ground_truth, **kwargs):
        return -self.KDE.log_density(ground_truth, state, weight)

class State(Module):

    def __init__(self):
      super().__init__()

    def forward(self, *, state: Tensor, **data):
        return state


class Weight(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *, weight: Tensor, **data):
        return weight