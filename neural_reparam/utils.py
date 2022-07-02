from typing import Union
import torch
import numpy as np
from collections.abc import Iterable

Array = Union[int, float, np.ndarray, torch.Tensor, Iterable]


def use_torch_with_numpy(func):
    def wrapper(array: Array):
        if isinstance(array, torch.Tensor):
            return func(array)
        elif isinstance(array, np.ndarray):
            out = func(torch.from_numpy(array))
            return out.numpy()
        elif isinstance(array, Iterable):
            return func(torch.Tensor(array))
        else:
            return func(torch.Tensor([array]))

    return wrapper


def use_torch_with_dual_numpy(func):
    def wrapper(array1: Array, array2: Array):
        if isinstance(array1, np.ndarray):
            array1 = torch.from_numpy(array1)
        if isinstance(array2, np.ndarray):
            array2 = torch.from_numpy(array1)
        result = func(array1, array2)
        if isinstance(result, torch.Tensor):
            return result.numpy()
        else:
            return result

    return wrapper


def stack_last_dim(*tensors: torch.Tensor) -> torch.Tensor:
    for tensor in tensors[1:]:
        assert tensor.shape == tensors[0].shape, "all must be same shape"
    if len(tensors[0].shape) == 1:
        return torch.stack(tensors, dim=-1)
    else:
        return torch.cat(tensors, dim=-1)
