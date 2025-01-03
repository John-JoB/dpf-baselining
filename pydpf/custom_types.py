from typing import Union, Callable, Any, Tuple, TypeAlias

from torch import Tensor


Aggregation : TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor, int], Tensor]
ImportanceKernelLikelihood : TypeAlias = Callable[[Tensor, Tensor, Tensor, int], Tuple[Tensor, Tensor, Tensor]]
ImportanceSamplerLikelihood : TypeAlias = Callable[[int, Tensor], Tuple[Tensor, Tensor, Tensor]]
WeightedSample : TypeAlias = Tuple[Tensor, Tensor]
ImportanceKernel : TypeAlias = Callable[[Tensor, Tensor, Tensor, int], WeightedSample]
ImportanceSampler : TypeAlias = Callable[[int, Tensor], WeightedSample]
Resampler : TypeAlias = [[Tensor, Tensor], WeightedSample]