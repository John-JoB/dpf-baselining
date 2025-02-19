import torch
from typing import Tuple, Callable
from types import FunctionType
from functools import update_wrapper

from warnings import warn
from torch import Tensor


def batched_select(tensor: Tensor, index: torch.LongTensor) -> Tensor:
    """
    Batched analog to tensor[index].
    Use to select along the mth dimension of tensor.
    index has a dimensions , tensor has b dimensions.
    a <= b
    The size of the every dimension but the last of index must be the same as the corresponding dimension in tensor.

    Parameters
    ----------
    tensor : A1 x A2 X ... Am x I x B1 X ... X Bn Tensor
        tensor to select from
    index : A1 x A2 x A3 X ... X Am X D torch.LongTensor
        tensor of indices to select from tensor A1.

    Returns
    -------
    output : A1 x A2 X ... X Am X D X B1 X ... X Bn Tensor
    """
    if tensor.dim() == 3 and index.dim() == 2:
        #Special case common case for efficiency
        return torch.gather(input = tensor, index = index.unsqueeze(-1).expand(-1, -1, tensor.size(2)), dim=1)
    elif tensor.dim() == index.dim():
        return torch.gather(input=tensor, index=index, dim=-1)
    elif tensor.dim() > index.dim():
        index_size = index.size()
        index_dim = index.dim()
        index = multiple_unsqueeze(index, tensor.dim() - index_dim)
        index = index.expand(*index_size,*tuple([tensor.size(i) for i in range(index_dim, tensor.dim())]))
        return torch.gather(input =tensor, index=index, dim=index_dim-1)
    raise ValueError('index cannot have more dimensions than tensor')

def back_index_reduce(tensor: Tensor, index:torch.LongTensor, default_value:torch.Tensor | float = 0., min_entries: int = 0, reduction = 'sum') -> Tensor:
    """
        Acts in the reverse direction to batched_select, applies a reduction over all elements in tensor with the same index.
        Use to select along the mth dimension of tensor.
        index has a dimensions , tensor has b dimensions.
        a <= b
        The size of the every dimension but the last of index must be the same as the corresponding dimension in tensor.

        Parameters
        ----------
        tensor : A1 x A2 X ... Am x D x  B1 x B2 X ... X Bn Tensor
            tensor to select from
        index : A1 x A2 x A3 X ... X Am X D torch.LongTensor
            tensor of indices to select from tensor A1.
        default_value : float | Tensor
            Default: 0. The default value to use for indices in output that are not present in the index. If default_value is a Tensor, it's size must be exactly equal to the output size
        min_entries : int
            Default: 0. The size of the indexed dimension in the output will always be the maximum of min_entries and the maximum value in index.
        reduction : str
            Default: 'sum'. The reduction to use, possible options are 'sum', 'prod', 'mean', 'amax', 'amin'.

        Returns
        -------
        output : A1 x A2 X ... X Am X I X B1 X ... X Bn Tensor
    """
    index_dim = index.dim()
    index_size = index.size()
    entries = torch.max(index).item()
    if min_entries is not None:
        entries = max(entries, min_entries)
    if entries != index_size[-1]:
        warn('Warning: backward pass only implemented if index has the same dimension as the output.\nConsider setting min_entries')
    if not isinstance(default_value, torch.Tensor):
        output = torch.full(size=(*tuple(index.size(i) for i in range(0, index_dim-1)), entries, *tuple(tensor.size(j) for j in range(index_dim, tensor.dim()))), fill_value=default_value, dtype=tensor.dtype, device=tensor.device)
    else:
        output = default_value
    index = multiple_unsqueeze(index, tensor.dim() - index_dim)
    index = index.expand(tensor.size())
    return torch.scatter_reduce(output, dim=index_dim-1, index=index, src=tensor, reduce=reduction, include_self=False)



def normalise(tensor: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """
    Normalise a log-space tensor to magnitude 1 along a specified dimension.
    Also return the norm.

    Parameters
    ----------
    tensor : Tensor
        tensor to normalise

    dim : int
     dimension to normalise along

    Returns
    -------
    norm_tensor: Tensor
        nomalised tensor

    norm: Tensor
        magnitude of tensor

    """
    norm = torch.logsumexp(tensor, dim=dim, keepdim=True)
    if torch.isinf(norm).any():
        raise ValueError('Row where all values -inf encountered')
    return tensor - norm, norm

def multiple_unsqueeze(tensor: Tensor, n: int, dim: int = -1) -> Tensor:
    """
    Unsqueeze multiple times at the same dimension.
    Equivalent to:

    for i in range(n):
        tensor = tensor.unsqueeze(d)
    return tensor

    Parameters
    ----------
    tensor : Tensor
        Tensor to unsqueeze
    n : int
        Number of times to unsqueeze the tensor
    dim : int
        Dimension to unsqueeze the tensor at

    Returns
    -------
    output : Tensor
        Unsqueezed tensor
    """
    if n == 0:
        return tensor
    if dim < 0:
        dim = tensor.dim() + dim
    return tensor[(slice(None),) * dim + (None, ) * n]


class doc_function:
    """
        Reflection hack to allow functions that only define a docstring.
    """

    def __init__(self, fun):
        """
            Mark an overriding function as docstring only. I.e. the function will default to it's parent's implementation at runtime.
        """
        self.fun = fun
        self.name = fun.__name__
        update_wrapper(self, fun)

    def update_function(self, instance, owner, overrides : FunctionType):
        setattr(owner, self.name, overrides)
        if instance is not None:
            return getattr(instance, self.name)
        #For static methods
        return getattr(owner, self.name)

    def __get__(self, instance, owner):
        if owner is None:
            raise RuntimeError(f'Attempting to use doc_function outside of a class.')
        bases = owner.__mro__
        for base in bases[1:]:
            try:
                overrides = base.__dict__[self.name]
                return self.update_function(instance, owner, overrides)
            except KeyError:
                pass
        raise AttributeError(f'Base classes have no attribute {self.name}.')

def MSE(prediction: Tensor, ground_truth: Tensor):
    return torch.sum(torch.mean((prediction - ground_truth) ** 2, dim=(0,1)))
"""
def add_kwargs(func):
   
    Modifies a function inplace to take extra optional keyword arguments as **kwargs.
    The input function must not have a '**' argument.
    If func is a torch.nn.Module (or a pydpf.Module) then the modification is on the forward() method.
    If func is any other non-function object then the modification is on the __call__() method.

    Parameters
    ----------
    func - Funtion to add **kwargs to.

    Returns
    -------
    None

    
    if isinstance(func, torch.nn.Module):
        try:
            func = func.__class__.forward
        except AttributeError:
            raise AttributeError(f"Module {func.__name__}  must implement the forward method.")
    elif not isinstance(func, FunctionType):
        try:
            if isinstance(func, type):
                func = func.__call__
            else:
                func = func.__class__.__call__
        except AttributeError:
            raise AttributeError(f"Object {func.__name__}  must be a function or implement the __call__ method.")
    code_obj = func.__code__
    #Raise error if function has a ** argument and it hasn't already been modified by this function.
    if code_obj.co_flags & 0x08:
        if code_obj.co_flags & 0x400:
            return
        raise AttributeError(f"Function {func.__name__} must not have a '**' argument.")
    old_args = list(code_obj.co_varnames)
    old_args.append('kwargs')
    new_code_obj = CodeType(code_obj.co_argcount,
                            code_obj.co_posonlyargcount,
                            code_obj.co_kwonlyargcount,
                            code_obj.co_nlocals + 1,
                            code_obj.co_stacksize,
                            code_obj.co_flags + 0x408,
                            code_obj.co_code,
                            code_obj.co_consts,
                            code_obj.co_names,
                            tuple(old_args),
                            code_obj.co_filename,
                            code_obj.co_name,
                            code_obj.co_qualname,
                            code_obj.co_firstlineno,
                            code_obj.co_linetable,
                            code_obj.co_exceptiontable,
                            code_obj.co_freevars,
                            code_obj.co_cellvars)
    func.__code__ = new_code_obj
    
"""