import contextlib
import gc
import os
from pathlib import Path
from typing import Callable, List, Union, Any

import numpy as np
import torch
import torch.optim


# noinspection PyAttributeOutsideInit
class AverageMeter:
    """Computes and stores the sum and the last value"""

    def __init__(self,
                 init_factory: Callable = None,
                 init_value: Any = 0.,
                 init_count=0):
        self.init_factory: Callable = init_factory
        self.init_value = init_value

        self.reset(init_count)

    def reset(self, init_count=0):
        if self.init_factory:
            self.last = self.init_factory()
            self.sum = self.init_factory()
        else:
            self.last = self.init_value
            self.sum = self.init_value
        self.count = init_count

    def update(self, value, n=1):
        self.last = value
        self.sum += value
        self.count += n

    def get_average(self):
        return self.sum / self.count


def static_vars(**kwargs):
    """ decorator to make static variables in function

    :param kwargs:
    :return:
    """
    def decorate(func: Callable):
        for k, a in kwargs.items():
            setattr(func, k, a)
        return func

    return decorate


def arr2str(a: np.ndarray, format_='e', ndigits=2) -> str:
    """convert ndarray of floats to a string expression.

    :param a:
    :param format_:
    :param ndigits:
    :return:
    """
    return np.array2string(
        a,
        formatter=dict(
            float_kind=(lambda x: f'{x:.{ndigits}{format_}}' if x != 0 else '0')
        )
    )


# deprecated. Use tqdm
def print_progress(iteration: int, total: int, prefix='', suffix='',
                   decimals=1, len_bar=0):
    percent = f'{100 * iteration / total:>{decimals + 4}.{decimals}f}'
    if len_bar == 0:
        len_bar = (min(os.get_terminal_size().columns, 80)
                   - len(prefix) - len(percent) - len(suffix) - 11)

    len_filled = len_bar * iteration // total
    bar = '#' * len_filled + '-' * (len_bar - len_filled)

    print(f'{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print('')


def print_cuda_tensors():
    """ Print all cuda Tensors """
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj)
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                print(type(obj), obj.size(), obj.device)
        finally:
            pass


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)
