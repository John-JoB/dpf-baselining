from torch import Tensor
import torch
from .base import Distribution
from ..base import Module, constrained_parameter, cached_property
from typing import Tuple, Union, Callable
from ..utils import doc_function
from warnings import warn

class MultivariateGaussian(Distribution):
    conditional = False

    half_log_2pi = (1/2) * torch.log(torch.tensor(2*torch.pi))

    def __init__(self, mean: Tensor, cholesky_covariance: Tensor, diagonal_cov:bool = False, gradient_estimator: str = 'reparameterisation', generator: Union[None, torch.Generator] = None) -> None:
        """
            A multivariate Gaussian distribution.

            Parameters
            ----------
            mean: Tensor
                1D tensor specifying the mean.
            cholesky_covariance: Tensor
                2D tensor specifying the (lower) Cholesky decomposition of the covariance matrix. If the upper triangular section has non-zero
                values these will be ignored.
            diagonal_cov: bool
                Whether to constrain the covariance to be diagonal. Default is False.
            gradient_estimator : str
                The gradient estimator to use, one of 'reparameterisation', 'score' or 'none'. Default is 'reparameterisation'.
            generator : Union[torch.Generator, None]
                The generator to control the rng when sampling kernels from the mixture.
        """

        super().__init__(gradient_estimator, generator)
        self.reparameterisable = True
        self.mean = mean
        self.cholesky_covariance_ = cholesky_covariance
        self.diagonal_cov = diagonal_cov
        self.dim = mean.size(0)
        if cholesky_covariance.device != mean.device:
            raise ValueError(f'Mean and Covariance should be on the same device, found {mean.device} and {cholesky_covariance.device}')
        if (cholesky_covariance.size(0) != self.dim) or (cholesky_covariance.size(1) != self.dim):
            raise ValueError(f'Covariance must have the same dimensions as the mean, found {self.dim} and {cholesky_covariance.size()}.')

    @constrained_parameter
    def cholesky_covariance(self) -> Tuple[Tensor, Tensor]:
        if self.diagonal_cov:
            diag = torch.diag(self.cholesky_covariance_.diag())
            return  self.cholesky_covariance_,  diag * diag.sign()
        tril = torch.tril(self.cholesky_covariance_)
        diag = tril.diagonal()
        diag.mul_(diag.sign())
        return self.cholesky_covariance_, tril

    @cached_property
    def inv_cholesky_cov(self):
        return torch.linalg.inv_ex(self.cholesky_covariance)[0]

    @cached_property
    def half_log_det_cov(self):
        return torch.linalg.slogdet(self.cholesky_covariance)[1]

    def _sample(self, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        if sample_size is None:
            true_sample_size = self.mean.size()
        else:
            true_sample_size = (*sample_size, self.mean.size(-1))
        output = torch.normal(0, 1, device=self.device, size= true_sample_size, generator=self.generator)
        output = self.mean + output @ self.cholesky_covariance.T
        return output


    @doc_function
    def sample(self, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        """
        Sample a Multivariate Gaussian distribution.

        Parameters
        ----------
        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. Draw a single sample without a sample dimension if None.

        Returns
        -------
        sample: Tensor
            A multivariate Gaussian sample.
        """
        pass

    def log_density(self, sample: Tensor) -> Tensor:
        """
        Returns the log density of a sample

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.
        """
        self.check_sample(sample)
        prefactor = -sample.size(-1) * MultivariateGaussian.half_log_2pi - self.half_log_det_cov
        residuals = sample - self.mean
        exponent = (-1/2) * torch.sum((residuals @ self.inv_cholesky_cov.T)**2, dim=-1)
        return prefactor + exponent

    def d_log_density(self, sample:Tensor) -> Tensor:
        self.check_sample(sample)
        residuals = sample - self.mean
        return -torch.einsum('...ki,...kj,...j->...i', self.inv_cholesky_cov, self.inv_cholesky_cov, residuals)

class _ConstCovGaussian(Distribution):

    conditional = True

    def _check_conditions(self, condition_on: Tensor) -> None:
        if condition_on.device != self.device:
            raise ValueError(f'condition_on should be on the same device as the distribution parameters, found {condition_on} and {self.device}.')

    def __init__(self, mean: Callable[[Tensor], Tensor], cholesky_covariance: Tensor, diagonal_cov, gradient_estimator: str = 'reparameterisation', generator: Union[None, torch.Generator] = None) -> None:
        super().__init__(gradient_estimator, generator)
        self.mean_fun = mean
        self.dim = cholesky_covariance.size(0)
        self.reparameterisable = True
        self.dist = MultivariateGaussian(torch.zeros((self.dim,), device=self.device), cholesky_covariance, diagonal_cov, gradient_estimator, generator)
        self.device = cholesky_covariance.device

    def _sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        self._check_conditions(condition_on)
        means = self.mean_fun(condition_on)
        batch_size = self.get_batch_size(means.size(), 1)
        if sample_size is None:
            sample = self.dist._sample(sample_size=batch_size)
        else:
            sample = self.dist._sample(sample_size=(*batch_size, *sample_size))
        return sample + self._unsqueeze_to_size(means, sample)

    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        self._check_conditions(condition_on)
        try:
            means = self.mean_fun(condition_on)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e}. \n This is likely to a mismatch in batch dimensions between the conditioning variables and the sample.')
        return self.dist.log_density(sample - self._unsqueeze_to_size(means, sample))

    def d_log_density(self, sample:Tensor, condition_on:Tensor) -> Tensor:
        self._check_conditions(condition_on)
        try:
            means = self.mean_fun(condition_on)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e}. \n This is likely to a mismatch in batch dimensions between the conditioning variables and the sample.')
        return self.dist.d_log_density(sample - self._unsqueeze_to_size(means, sample))

class _GeneralCovGaussian(Distribution):
    conditional = True

    def _check_conditions(self, condition_on: Tensor) -> None:
        if condition_on.device != self.device:
            raise ValueError(f'condition_on should be on the same device as the distribution parameters, found {condition_on.device} and {self.device}.')

    def __init__(self,
                mean: Callable[[Tensor], Tensor],
                cholesky_covariance: Callable[[Tensor], Tensor],
                gradient_estimator: str = 'reparameterisation',
                force_diagonal_cov: bool = False,
                dim: Union[int, None] = None,
                device: Union[torch.device, None] = None,
                generator: Union[torch.Generator, None] = None
                ):

        super().__init__(gradient_estimator, generator)
        self.mean_fun = mean
        self.cov_fun = cholesky_covariance
        self.dim = dim
        self.reparameterisable = True
        self.dist = MultivariateGaussian(torch.zeros((self.dim,), device=self.device), torch.eye(self.dim), True, gradient_estimator, generator)
        self.device = device
        self.force_diagonal = force_diagonal_cov

    def prepare_cov(self, cov: Tensor) -> Tensor:
        diagonal = torch.abs(torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1)))
        tril = torch.tril(cov, diagonal=-1)
        return diagonal + tril

    def _sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        self._check_conditions(condition_on)
        means = self.mean_fun(condition_on)
        cov = self.cov_fun(condition_on)

        batch_size = self.get_batch_size(means.size(), 1)
        if sample_size is None:
            sample = torch.normal(0, 1, device=self.device, size=batch_size, generator=self.generator)
        else:
            sample = torch.normal(0, 1, device=self.device, size=(*batch_size, *sample_size), generator=self.generator)

        means = self._unsqueeze_to_size(means, sample)
        if self.force_diagonal:
            cov = self._unsqueeze_to_size(cov, sample.dim(), 1)
            return sample*cov + means

        cov = self.prepare_cov(cov)
        cov = self._unsqueeze_to_size(cov, sample.dim() + 1, 2)
        return sample.unsqueeze(-2) @ cov  + means

    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        self._check_conditions(condition_on)
        means = self.mean_fun(condition_on)
        means = self._unsqueeze_to_size(means, sample)
        cov = self.cov_fun(condition_on)

        residuals = sample - means
        if self.force_diagonal:
            cov = self._unsqueeze_to_size(cov, sample, 1)
            half_log_det_cov = torch.log(torch.prod(cov, dim=-1))
            residuals_cov = residuals/cov
        else:
            cov = self.prepare_cov(cov)
            cov = self._unsqueeze_to_size(cov, sample.dim() + 1, 2)
            half_log_det_cov = torch.linalg.slogdet(cov)[1]
            residuals_cov = torch.linalg.solve_triangular(cov, residuals.unsqueeze(-1))


        prefactor =  -sample.size(-1) * MultivariateGaussian.half_log_2pi - half_log_det_cov

        exponent = (-1 / 2) * torch.sum(residuals_cov ** 2, dim=-1)
        return prefactor + exponent


    def d_log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        self._check_conditions(condition_on)
        means = self.mean_fun(condition_on)
        means = self._unsqueeze_to_size(means, sample)
        root_cov = self.cov_fun(condition_on)

        residuals = sample - means
        if self.force_diagonal:
            root_cov = self._unsqueeze_to_size(root_cov, sample, 1)
            return -residuals/(root_cov**2)
        else:
            root_cov = self.prepare_cov(root_cov)
            root_cov = self._unsqueeze_to_size(root_cov, sample.dim() + 1, 2)
            inv_cov = torch.linalg.inv(root_cov)
            return -torch.einsum('...ki,...kj,...j->...i', inv_cov, inv_cov, residuals)




class ConditionalGaussian(Distribution):
    """
        Class to dispatch to different Gaussian implementations.
    """

    def __new__(cls, mean: Union[Callable[[Tensor], Tensor], Tensor],
                cholesky_covariance: Union[Callable[[Tensor], Tensor], Tensor],
                force_diagonal_cov: bool = False,
                dim: Union[int, None] = None,
                device: Union[torch.device, None] = None,
                gradient_estimator: str = 'reparameterisation',
                generator: Union[torch.Generator, None] = None,
                ) -> Distribution:

        if isinstance(mean, Tensor) and isinstance(cholesky_covariance, Tensor):
            warn('Using the conditional API to create a non-conditional distribution, are you sure this is correct?')
            return MultivariateGaussian(mean, cholesky_covariance, force_diagonal_cov, gradient_estimator, generator)

        if isinstance(cholesky_covariance, Tensor):
            return _ConstCovGaussian(mean, cholesky_covariance, force_diagonal_cov, gradient_estimator, generator)

        if isinstance(mean, Tensor):
            return _GeneralCovGaussian(lambda t: mean, cholesky_covariance, force_diagonal_cov, mean.size(-1), mean.device, gradient_estimator, generator)

        if dim is None:
            raise ValueError('Dimension must be specified if both the mean and the covariance are functions')
        if device is None:
            raise ValueError('Dimension must be specified if both the mean and the covariance are functions')

        return _GeneralCovGaussian(mean, cholesky_covariance, force_diagonal_cov, dim, device, gradient_estimator, generator)


    @doc_function
    def __init__(self, mean: Union[Callable[[Tensor], Tensor], Tensor],
                cholesky_covariance: Union[Callable[[Tensor], Tensor], Tensor],
                force_diagonal_cov: bool = False,
                dim: Union[int, None] = None,
                device: Union[torch.device, None] = None,
                gradient_estimator: str = 'reparameterisation',
                generator: Union[torch.Generator, None] = None,
                ):
        """
        A general conditional Gaussian distribution, where the mean and covariance can be given as arbitrary functions of some conditioning tensor.

        Both the mean and cholesky_covariance can be either a Tensor or a function from a Tensor to a Tensor.

        If the mean is a function, it should take an arbitrary conditioning Tensor and output a BxD Tensor where the mean vectors are along the last dimension, and B is zero or more batch dimensions.

        If cholesky_covariance is a fuction and force_diagonal is False, it should take an arbitrary conditioning Tensor and output a BxDxD Tensor where the lower triangular cholesky covariance matricies
        are the last two dimension, and B is zero or more batch dimensions. If the matrix is not a valid lower triangular form it will be mapped as so by setting the diagonal to be positive and all elements above the diagonal to be zero.

        If cholesky_covariance is a fuction and force_diagonal is True, it should take an arbitrary conditioning Tensor and output a BxD Tensor where the last dimension contains the standard deviations of the sample dimensions,
        and B is zero or more batch dimensions.

        Notes
        ------
        In the most general case, repeatedly computing large batches of matrix determinants and inverses is slow. We implement optimised routines for the special cases that the covariance is constant or that it is diagonal.
        If the user's problem does not fit these cases but has other structure that can be taken advantage of it is recommended that they implement a custom sampling proceedure rather than use this.

        Parameters
        ----------
        mean: Union[Callable[[Tensor], Tensor], Tensor]
            The means or a function to calculate them
        cholesky_covariance: Union[Callable[[Tensor], Tensor], Tensor]
            The lower cholesky decomposition of the covariance, or a function to calculate it.
        force_diagonal_cov: bool
            Whether to force the covariance matrix to be diagonal or not.
        dim: Union[int, None]
            If both the mean and the covariance are given as functions, then the dimension of the distribution must be provided, otherwise it is inferred from the constant parameters.
        device : Union[torch.device, None]
            If both the mean and the covariance are given as functions, then the device of the distribution must be provided, otherwise it is inferred from the constant parameters.
        gradient_estimator : str
                The gradient estimator to use, one of 'reparameterisation', 'score' or 'none'.
        generator : Union[torch.Generator, None]
            The generator to control the rng when sampling kernels from the mixture.
        """

        super().__init__(gradient_estimator, generator)

    @doc_function
    def sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        """
        Sample a conditional Gaussian distribution.

        Parameters
        ----------
        condition_on: Tensor
            The vector to condition the distribution on.

        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            A sample of a conditional Gaussian, conditioned on condition_on.

        """
        pass

    def _sample(self, *args, **kwargs) -> Tensor:
        pass

    @doc_function
    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        condition_on: Tensor
            The vector to condition the distribution on.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.

        """
        pass

class LinearGaussian(Distribution):
    conditional = True

    def __new__(cls,
                weight: Tensor,
                bias: Tensor,
                cholesky_covariance: Tensor,
                diagonal_cov: bool = False,
                constrain_spectral_radius: Union[float, None] = None,
                gradient_estimator: str = 'reparameterisation',
                generator: Union[torch.Generator, None] = None):

        class LinearTransform(Module):
            def __init__(self):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return (self.constrained_weight @ x.unsqueeze(-1)).squeeze() + self.bias

            @constrained_parameter
            def constrained_weight(self):
                if constrain_spectral_radius is not None:
                   # print('eigvals')
                    #print(self.weight)
                    eigvals = torch.linalg.eigvals(self.weight)
                    #print('radius')
                    spectral_radius = torch.max(torch.abs(eigvals))
                    if  spectral_radius > constrain_spectral_radius:
                        return self.weight, self.weight / spectral_radius
                return self.weight, self.weight

        dim = weight.size(-2)
        device = weight.device
        if (cholesky_covariance.size(0) != dim) or (cholesky_covariance.size(1) != dim):
            raise ValueError(f'Covariance must have the same dimensions as the weights first dimension, found {cholesky_covariance.size()} and {weight.size()}.')
        if bias.size(0) != dim:
            raise ValueError(f'Bias must have the same dimensions as the weights first dimension, found {bias.size()} and {weight.size()}')
        if cholesky_covariance.device != device:
            raise ValueError(f'Weight and Covariance should be on the same device, found {device} and {cholesky_covariance.device}')
        if cholesky_covariance.device != device:
            raise ValueError(f'Weight and bias should be on the same device, found {device} and {bias.device}')

        return _ConstCovGaussian(LinearTransform(), cholesky_covariance, diagonal_cov, gradient_estimator, generator)

    @doc_function
    def __init__(self,
                weight: Tensor,
                bias: Tensor,
                cholesky_covariance: Tensor,
                diagonal_cov: bool = False,
                constrain_spectral_radius: Union[float, None] = None,
                gradient_estimator: str = 'reparameterisation',
                generator: Union[torch.Generator, None] = None):
        """
            A Gaussian conditional distribution, where the means of the Gaussian are conditional on a supplied variable, X, through the linear map WX + B. Where W is the weights and B is the bias.

            Parameters
            ----------
            weight: Tensor
                2D tensor specifying the weight matrix.
            bias: Tensor
                1D tensor specifying the bias.
            cholesky_covariance: Tensor
                2D tensor specifying the (lower) Cholesky decomposition of the covariance matrix. If the upper triangular section has non-zero
                values these will be ignored.
            diagonal_cov: bool
                Whether to force the covariance matrix to be diagonal or not.
            constrain_spectral_radius: Union[int, None]
                If constrain_spectral_radius is an integer, then the weight matrix will be scaled so that it's spectral radius never exceed the passed value. If constrain_spectral_radius is None, no scaling is applied.
            gradient_estimator : str
                The gradient estimator to use, one of 'reparameterisation', 'score' or 'none'.
            generator : Union[torch.Generator, None]
                The generator to control the rng when sampling kernels from the mixture.
        """

    @doc_function
    def sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        """
        Sample a multivariate Linear Gaussian.
        The means of the Gaussian are calculated as condition_on @ self.weight + self.bias.

        Parameters
        ----------
        condition_on: Tensor
            The vector to condition the distribution on.

        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            A sample of a multivariate Linear Gaussian, conditioned on condition_on.

        """
        pass

    def _sample(self, *args, **kwargs) -> Tensor:
        pass

    @doc_function
    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.
        The means of the Gaussian are calculated as condition_on @ self.weight + self.bias.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        condition_on: Tensor
            The vector to condition the distribution on.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.

        """
        pass