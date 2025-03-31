
from .base import Distribution
from typing import Union, Tuple, Callable
import torch
from torch import Tensor
from ..utils import multiple_unsqueeze, doc_function

class DeltaMeasure(Distribution):
    def __init__(self, gradient_estimator: str, generator: Union[torch.Generator, None], loc: Tensor) -> None:
        super().__init__(gradient_estimator, generator)
        self.loc = loc
        self.dim = loc.size(-1)

    def _sample(self, sample_size: Union[Tuple[int, ...], None] = None):
        if sample_size is None:
            return self.loc.clone()
        sample = multiple_unsqueeze(self.loc, len(sample_size), 0)
        return sample.expand((*sample_size, -1)).clone()

    @doc_function
    def sample(self, sample_size: Union[Tuple[int, ...], None] = None):
        """
        Sample a delta distribution. I.e. set the output to the centre of the distribution.

        Parameters
        ----------
        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. Draw a single sample without a sample dimension if None.

        Returns
        -------
        sample: Tensor
            A delta distributed sample.
        """
        pass

    def log_density(self, sample) -> Tensor:
        """
        Log density of the sample. Log density is not well-defined for the delta distribution.
        But it is useful to output 0 when the sample is at loc, and -infinity when not.
        This has the effect of ignoring the delta-distributed/deterministic components when combined with other distributions.
        It is intended that the user should be careful about using this distribution in cases where the sample might not be exactly on the loc.
        A small numerical error is permitted.

        Parameters
        ----------
        sample: Tensor
        The delta distributed sample.

        Returns
        -------
        log_density: Tensor
        The log-density of the sample.
        """
        unsqueeze_loc = self._unsqueeze_to_size(self.loc, sample, 1)
        diff = torch.max(sample - unsqueeze_loc, dim = -1)[0]
        return torch.where(torch.abs(diff) < 1e-8, torch.zeros_like(diff), torch.full_like(diff, -torch.inf))

class ConditionalDelta(Distribution):
    def __init__(self, dim:int, gradient_estimator: str, generator: Union[torch.Generator, None], loc_fun: Callable[[Tensor], Tensor] = lambda x:x) -> None:
        super().__init__(gradient_estimator, generator)
        self.fun = loc_fun
        self.dim = dim

    def _sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None):
        locs = self.fun(condition_on)
        if sample_size is None:
            return locs
        sample = multiple_unsqueeze(locs, len(sample_size), -2)
        return sample.expand((*locs.size(), *sample_size, -1)).clone()

    @doc_function
    def sample(self, sample_size: Union[Tuple[int, ...], None] = None):
        """
        Sample a conditional delta distribution. I.e. set the output to the centre of the distribution.

        Parameters
        ----------
        condition_on:
            The tensor to condition on.

        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. Draw a single sample without a sample dimension if None.

        Returns
        -------
        sample: Tensor
            A multivariate Gaussian sample.
        """
        pass

    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        """
        Log density of the sample. Log density is not well-defined for the delta distribution.
        But it is useful to output 0 when the sample is at loc, and -infinity when not.
        This has the effect of ignoring the delta-distributed/deterministic components when combined with other distributions.
        It is intended that the user should be careful about using this distribution in cases where the sample might not be exactly on the loc.
        A small numerical error is permitted.

        Parameters
        ----------
        sample: Tensor
        The delta distributed sample.

        Returns
        -------
        log_density: Tensor
        The log-density of the sample.
        """
        locs = self.fun(condition_on)
        unsqueeze_loc = self._unsqueeze_to_size(locs, sample, 1)
        diff = torch.max(sample - unsqueeze_loc, dim = -1)[0]
        return torch.where(torch.abs(diff) < 1e-8, torch.zeros_like(diff), torch.full_like(diff, -torch.inf))