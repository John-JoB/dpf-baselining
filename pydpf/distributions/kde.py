from .base import Distribution
from torch import Tensor, dtype
from .Gaussian import MultivariateGaussian
from typing import Iterable, Union, Tuple, List
import torch

from .delta import DeltaMeasure
from ..utils import doc_function, batched_select
from ..resampling import MultinomialResampler


class CompoundDistribution(Distribution):
    """
        Class for when the desired random variable can be expressed as a set of independent components.
        Primarily included to allow kernels that have different distributions along different dimensions.
    """
    conditional = False

    def __init__(self, distributions: Iterable[Distribution], gradient_estimator: str, generator: Union[torch.Generator, None]):
        """
        A wrapper that concatenates several distributions into a single distribution object.
        Interdependence between the distributions is not permitted.

        Parameters
        ----------
        distributions: Iterable[Distribution]
            An iterable of the distributions that form the components of the compound.
        gradient_estimator : str
            The gradient estimator to use, one of 'reparameterisation', 'score' or 'none'.
        generator : Union[torch.Generator, None]
            The generator to control the rng when sampling kernels from the mixture.
        """
        super().__init__(gradient_estimator, generator)
        self.dists = distributions
        self.reparameterisable = True
        self.dim = 0
        #register submodules
        self.dists = torch.nn.ModuleList(distributions)
        for dist in self.dists:
            self.dim += dist.dim
            self.reparameterisable = self.reparameterisable and dist.reparameterisable
            if type(dist).conditional:
                raise TypeError(f'None of the component distributions may be conditional, detected {type(dist)} which is.')
            if self.device != dist.device:
                raise ValueError(f'All component distributions must have all parameters on the same device, found {self.device} and {dist.device}.')


    @doc_function
    def sample(self, sample_size: Union[Tuple[int], None]) -> Tensor:
        """
        Sample a Compound distribution.
        The sample is the concatenation of samples from the components distributions along the last axis.

        Parameters
        ----------
        sample_size : Union[Tuple[int], None]
            The size of the sample to draw. If None then a single sample is drawn and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            The resulting sample.
        """
        pass

    def _sample(self, sample_size: Union[Tuple[int], None]) -> Tensor:
        samples = []
        for dist in self.dists:
            samples.append(dist.sample(sample_size))
        return torch.cat(samples, dim=-1)

    def log_density(self, sample: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.

        """
        output = None
        dim_count = 0
        for dist in self.dists:
            if output is None:
                output = dist.log_density(sample[..., dim_count:dim_count+dist.dim])
            else:
                output += dist.log_density(sample[..., dim_count:dim_count+dist.dim])
            dim_count += dist.dim
        return output


def sample_only_multinomial(state: Tensor, weights: Tensor, generator) -> Tensor:
    with torch.no_grad():
        sampled_indices = torch.multinomial(torch.exp(weights), weights.size(1), replacement=True, generator=generator).detach()
    return batched_select(state, sampled_indices)


class KernelMixture(Distribution):
    '''
        Class for KDE mixtures.
    '''
    conditional = True

    def make_kernel(self, kernel: str, dim: int) -> Distribution:
        if kernel == 'Gaussian':
            cov = torch.nn.Parameter(torch.eye(dim, device=self.generator.device) * torch.rand(dim, device=self.generator.device, generator=self.generator))
            return MultivariateGaussian(torch.zeros(dim, device=self.generator.device), cov, generator=self.generator, gradient_estimator='none', diagonal_cov=True)
        if kernel == 'Delta':
            return DeltaMeasure(gradient_estimator='none', generator=self.generator, loc=torch.zeros((dim,), device=self.generator.device, dtype = torch.float32))
        raise NotImplementedError('Only Gaussian and Delta kernels are implemented for automatic generation.')

    def __init__(self, kernel: Union[List[Tuple[str, int]]], gradient_estimator: str, generator: Union[torch.Generator, None]):
        """
        Create a kernel density mixture.
        The parameter kernel is an unconditional distribution which will be convolved over the kernel density mixture.
        The resultant distribution is conditional on the locations and weights of the kernels.

        Notes
        -----
        The parameter kernel can either be a valid KernelMixture or a recipe for creating one. The recipe is specified by a list of 2-element Tuples. The first element is the name of the distribution and
        the second is the number of dimensions that should be distributed as the given distribution. For example if the state was the position and orientation of an object in 2D, where the positions have a Gaussian Kernel,
        and the orientation a von Mises kernel, the appropriate list would be [('Gaussian', 2), ('von Mises', 1)].

        Parameters
        ----------
        kernel: Union[List[Tuple[str, int]], KernelMixture
            The kernel to convolve over the particles to form the KDE sampling distribution.
        gradient_estimator : str
            The gradient estimator to use, one of 'reparameterisation', 'score' or 'none'.
        generator : Union[torch.Generator, None]
            The generator to control the rng when sampling kernels from the mixture.
        """


        super().__init__(gradient_estimator, generator)
        self.resampler = MultinomialResampler(generator=generator)
        if isinstance(kernel, Distribution):
            self.kernel = kernel
        elif len(kernel) == 1:
            self.kernel = self.make_kernel(kernel[0][0], kernel[0][1])
        else:
            subkernels = [self.make_kernel(subkernel[0], subkernel[1]) for subkernel in kernel]
            self.kernel = CompoundDistribution(subkernels, gradient_estimator='none', generator=generator)
        self.reparameterisable = True
        self.dim = self.kernel.dim
        if type(self.kernel).conditional:
            raise ValueError(f'The kernel distribution cannot be conditional, detected {type(self.kernel)} which is.')

    def _check_conditions(self, loc: Tensor, weight: Tensor) -> None:
        if loc.device != self.device:
            raise ValueError(f'loc must be on the same device as the distribution, found {loc.device} and {self.device}.')
        if weight.device != self.device:
            raise ValueError(f'weight must be on the same device as the distribution, found {weight.device} and {self.device}.')
        if loc.size(-1) != self.dim:
            raise ValueError(f'It is not permitted for the kernel to have a different dimension as the space it is convolved over, found {loc.size(-1)} and {self.dim}.')
        if weight.size(-1) != loc.size(-2):
            raise ValueError(f'Differing number of kernels locations{loc.size(-2)} and weights {weight.size(-1)}.')

    @doc_function
    def sample(self,  loc: Tensor, weight: Tensor, sample_size: Union[Tuple[int], None]) -> Tensor:
        """
        Sample a KDE mixture

        Parameters
        ----------
        loc : Tensor
            Locations of the Kernels

        weight  : Tensor
            Weights of the Kernels

        sample_size : Union[Tuple[int], None]
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        Sample: Tensor
         A sample from the KDE mixture.
        """
        pass

    def _sample(self,  loc: Tensor, weight: Tensor, sample_size: Union[Tuple[int], None]) -> Tensor:
        #Multinomial resampling is sampling from a KDE with a dirac kernel.
        self._check_conditions(loc, weight)
        try:
            sampled_locs, _ = self.resampler(loc, weight, generator=self.generator)
        except Exception as e:
            raise RuntimeError(f'Failed to sample kernels with error: \n {e} \n This is likely due to a mismatch in batch dimensions.')
        batch_size = self.get_batch_size(sampled_locs.size(), 2)
        if sample_size is None:
            sample = self.kernel._sample(sample_size=batch_size)
        else:
            sample = self.kernel._sample(sample_size=(*batch_size, *sample_size))
        return sampled_locs + sample

    def log_density(self, sample: Tensor, loc: Tensor, weight: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        loc : Tensor
            Locations of the Kernels

        weight  : Tensor
            Weights of the Kernels.

        Returns
        -------
        Sample: Tensor
         The log density of each datum in the sample.
        """

        try:
            self._check_conditions(loc, weight)
            densities = self.kernel.log_density(sample.unsqueeze(-2) - self._unsqueeze_to_size(loc, sample.unsqueeze(-2), 2))
            return torch.logsumexp(densities + weight.unsqueeze(-2), dim=-1)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e} \n This is likely to a mismatch in batch dimensions.')