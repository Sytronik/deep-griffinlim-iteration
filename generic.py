"""
Generic type & functions for torch.Tensor and np.ndarray
"""
from typing import Sequence, TypeVar, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

TensArr = TypeVar('TensArr', Tensor, ndarray)
TensArrOrSeq = Union[TensArr, Sequence[TensArr]]

dict_package = {Tensor: torch, ndarray: np}
dict_cat_stack_fn = {(Tensor, 'cat'): torch.cat,
                     (ndarray, 'cat'): np.concatenate,
                     (Tensor, 'stack'): torch.stack,
                     (ndarray, 'stack'): np.stack,
                     }


class DataPerDevice:
    __slots__ = ('data',)

    def __init__(self, data_np: ndarray):
        self.data = {ndarray: data_np}

    def __getitem__(self, typeOrtup):
        if type(typeOrtup) == tuple:
            _type, device = typeOrtup
        elif typeOrtup == ndarray:
            _type = ndarray
            device = None
        else:
            raise IndexError

        if _type == ndarray:
            return self.data[ndarray]
        else:
            if typeOrtup not in self.data:
                self.data[typeOrtup] = convert(self.data[ndarray],
                                               Tensor,
                                               device=device)
            return self.data[typeOrtup]

    def get_like(self, other: TensArr):
        if type(other) == Tensor:
            return self[Tensor, other.device]
        else:
            return self[ndarray]

def convert(a: TensArr, astype: type, device: Union[int, torch.device] = None) -> TensArr:
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, device=device)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)
