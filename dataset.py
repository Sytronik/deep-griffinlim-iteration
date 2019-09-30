import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, Union, Optional

import numpy as np
import scipy.io as scio
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

# from generic import DataPerDevice, TensArr
from hparams import Channel, hp

TensArr = Union[ndarray, Tensor]
StrOrSeq = TypeVar('StrOrSeq', str, Sequence[str])
TupOrSeq = TypeVar('TupOrSeq', tuple, Sequence[tuple])
DataDict = Dict[str, Any]


def xy_signature(func):
    def wrapper(self, *args, **kwargs):
        assert (len(args) > 0) ^ (len(kwargs) > 0)
        if len(args) == 2:
            kwargs['x'] = args[0]
            kwargs['y'] = args[1]
        elif len(args) == 1:
            kwargs['x'] = args[0]

        output: dict = func(self, **kwargs)
        if len(output) == 2:
            return output['x'], output['y']
        else:
            return output.popitem()[1]

    return wrapper


# class Normalization:
#     """
#     Calculating and saving mean/std of all mel spectrogram with respect to time axis,
#     applying normalization to the spectrogram
#     This is need only when you don't load all the data on the RAM
#     """

#     @staticmethod
#     def _sum(a: ndarray) -> ndarray:
#         return LogModule.log_(a).sum(axis=1, keepdims=True)

#     @staticmethod
#     def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
#         return ((LogModule.log_(a) - mean_a)**2).sum(axis=1, keepdims=True)

#     @staticmethod
#     def _load_data(fname: Union[str, Path], key: str, queue: mp.Queue) -> None:
#         x = np.load(fname, mmap_mode='r')[key].astype(np.float32, copy=False)
#         if hp.feature == 'mulspec' and key == hp.spec_data_names['x']:
#             x = x[..., :x.shape[-1]//2]
#         queue.put(x)

#     @staticmethod
#     def _calc_per_data(data,
#                        list_func: Sequence[Callable],
#                        args: Sequence = None,
#                        ) -> Dict[Callable, Any]:
#         """ gather return values of functions in `list_func`

#         :param list_func:
#         :param args:
#         :return:
#         """

#         if args:
#             result = {f: f(data, arg) for f, arg in zip(list_func, args)}
#         else:
#             result = {f: f(data) for f in list_func}
#         return result

#     def __init__(self, mean, std):
#         self.mean = DataPerDevice(mean.astype(np.float32, copy=False))
#         self.std = DataPerDevice(std.astype(np.float32, copy=False))

#     @classmethod
#     def calc_const(cls, all_files: List[Path], key: str):
#         """

#         :param all_files:
#         :param key: data name in npz file
#         :rtype: Normalization
#         """

#         # Calculate summation & size (parallel)
#         list_fn = (np.size, cls._sum)
#         pool_loader = mp.Pool(hp.num_disk_workers)
#         pool_calc = mp.Pool(min(mp.cpu_count() - hp.num_disk_workers - 1, 6))
#         with mp.Manager() as manager:
#             queue_data = manager.Queue()
#             pool_loader.starmap_async(cls._load_data,
#                                       [(f, key, queue_data) for f in all_files])
#             result: List[mp.pool.AsyncResult] = []
#             for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
#                 data = queue_data.get()
#                 result.append(pool_calc.apply_async(
#                     cls._calc_per_data,
#                     (data, list_fn)
#                 ))

#         result: List[Dict] = [item.get() for item in result]

#         sum_size = np.sum([item[np.size] for item in result])
#         sum_ = np.sum([item[cls._sum] for item in result],
#                       axis=0, dtype=np.float32)
#         mean = sum_ / (sum_size // sum_.size)

#         # Calculate squared deviation (parallel)
#         with mp.Manager() as manager:
#             queue_data = manager.Queue()
#             pool_loader.starmap_async(cls._load_data,
#                                       [(f, key, queue_data) for f in all_files])
#             result: List[mp.pool.AsyncResult] = []
#             for _ in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
#                 data = queue_data.get()
#                 result.append(pool_calc.apply_async(
#                     cls._calc_per_data,
#                     (data, (cls._sq_dev,), (mean,))
#                 ))

#         pool_loader.close()
#         pool_calc.close()
#         result: List[Dict] = [item.get() for item in result]
#         print()

#         sum_sq_dev = np.sum([item[cls._sq_dev] for item in result],
#                             axis=0, dtype=np.float32)

#         std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)

#         return cls(mean, std)

#     def astuple(self):
#         return self.mean.data[ndarray], self.std.data[ndarray]

#     # normalize and denormalize functions can accept a ndarray or a tensor.
#     def normalize(self, a: TensArr) -> TensArr:
#         return ((a - self.mean.get_like(a)[..., -a.shape[-1]:])
#                 / (2 * self.std.get_like(a)[..., -a.shape[-1]:]))

#     def normalize_(self, a: TensArr) -> TensArr:  # in-place version
#         a -= self.mean.get_like(a)[..., -a.shape[-1]:]
#         a /= 2 * self.std.get_like(a)[..., -a.shape[-1]:]

#         return a

#     def denormalize(self, a: TensArr) -> TensArr:
#         return (a * (2 * self.std.get_like(a)[..., -a.shape[-1]:])
#                 + self.mean.get_like(a)[..., -a.shape[-1]:])

#     def denormalize_(self, a: TensArr) -> TensArr:  # in-place version
#         a *= 2 * self.std.get_like(a)[..., -a.shape[-1]:]
#         a += self.mean.get_like(a)[..., -a.shape[-1]:]

#         return a

#     def __str__(self):
#         return f'mean - {self.mean[ndarray].shape}, std - {self.std[ndarray].shape}'


class ComplexSpecDataset(Dataset):
    """ Directional Spectrogram Dataset

    In the `split` class method, following member variables will be copied or split.
    (will be copied)
    _PATH
    _needs
    _trannorm

    (will be split)
    _all_files
    """

    def __init__(self, kind_data: str,
                 **kwargs: Channel):
        self._PATH = hp.dict_path[f'feature_{kind_data}']

        # default needs
        self._needs = dict(x=Channel.ALL, y=Channel.ALL,
                           y_mag=Channel.ALL,
                           length=Channel.ALL,
                           path_speech=Channel.ALL)
        self.set_needs(**kwargs)

        self._all_files = [f for f in self._PATH.glob('*.*') if hp.is_featurefile(f)]
        self._all_files = sorted(self._all_files)
        if hp.n_data > 0:
            self._all_files = self._all_files[:hp.n_data]

        # self.norm_modules = dict()
        # if kind_data == 'train':
        #     # path_normconst: path of the file that has information about mean, std, ...
        #     path_normconst = hp.dict_path[f'normconst_{kind_data}']

        #     if not hp.refresh_const and path_normconst.exists():
        #         # when normconst file exists
        #         npz_normconst = np.load(path_normconst, allow_pickle=True)
        #         self.norm_modules['x'] = Normalization(*npz_normconst['normconst_x'])
        #         self.norm_modules['y'] = Normalization(*npz_normconst['normconst_y'])
        #     else:
        #         print('calculate normalization consts for input')
        #         self.norm_modules['x'] \
        #             = Normalization.calc_const(self._all_files, key=hp.spec_data_names['x'])
        #         print('calculate normalization consts for output')
        #         self.norm_modules['y'] \
        #             = Normalization.calc_const(self._all_files, key=hp.spec_data_names['y'])
        #         np.savez(path_normconst,
        #                  normconst_x=self.norm_modules['x'].astuple(),
        #                  normconst_y=self.norm_modules['y'].astuple())
        #         scio.savemat(path_normconst.with_suffix('.mat'),
        #                      dict(normconst_x=self.norm_modules['x'].astuple(),
        #                           normconst_y=self.norm_modules['y'].astuple()))

        #     print(f'normalization consts for input: {self.norm_modules["x"]}')
        #     print(f'normalization consts for output: {self.norm_modules["y"]}')
        # else:
        #     assert 'x' in norm_modules and 'y' in norm_modules
        #     self.norm_modules = norm_modules

        print(f'{len(self._all_files)} files are prepared from {kind_data.upper()}.')

    def __len__(self):
        return len(self._all_files)

    def __getitem__(self, idx: int) -> DataDict:
        """

        :param idx:
        :return: DataDict
            Values can be an integer, ndarray, or str.
        """
        sample = dict()
        with np.load(self._all_files[idx], mmap_mode='r') as npz_data:
            for k, v in self._needs.items():
                if v.value:
                    data: ndarray = npz_data[hp.spec_data_names[k]]
                    if type(data) == np.str_:
                        sample[k] = str(data)
                    elif data.size <= 1:
                        sample[k] = int(data.item())
                    else:
                        # F, T, C
                        if data.dtype == np.complex64:
                            assert data.flags['C_CONTIGUOUS']
                            data = data.view(dtype=np.float32).reshape((*data.shape, 2))
                        sample[k] = torch.from_numpy(data)

        for xy in ('x', 'y'):
            sample[f'T_{xy}'] = sample[xy].shape[1]

        return sample

    @torch.no_grad()
    def pad_collate(self, batch: List[DataDict]) -> DataDict:
        """ return data with zero-padding

        Important data like x, y are all converted to Tensor(cpu).
        :param batch:
        :return: DataDict
            Values can be an Tensor(cpu), list of str, ndarray of int.
        """
        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted].tolist()
        T_ys = [batch[idx].pop('T_y') for idx in idxs_sorted]
        length = [batch[idx].pop('length') for idx in idxs_sorted]

        result['T_xs'], result['T_ys'], result['length'] = T_xs, T_ys, length

        for key, value in batch[0].items():
            if type(value) == str:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                if len(batch) > 1:
                    # B, T, F, C
                    data = [batch[idx][key].permute(1, 0, 2) for idx in idxs_sorted]
                    data = pad_sequence(data, batch_first=True)
                    # B, C, F, T
                    data = data.permute(0, 3, 2, 1)
                else:  # B, C, F, T
                    data = batch[0][key].unsqueeze(0).permute(0, 3, 1, 2)

                # if key == 'x' or key == 'y':
                #     if hp.feature == 'mulspec' and key == 'x':
                #         C = data.shape[-1]
                #         normalized = torch.cat(
                #             (self.normalize(**{key: data[..., :C // 2]}),
                #              data[..., C // 2:] / np.sqrt(np.pi**2 / 3)),
                #             dim=-1,
                #         )
                #     else:
                #         normalized = self.normalize(**{key: data})
                #     normalized = normalized.permute(0, -1, -3, -2).contiguous()
                #     result[f'normalized_{key}'] = normalized

                result[key] = data.contiguous()

        return result

    @staticmethod
    @torch.no_grad()
    def decollate_padded(batch: DataDict, idx: int) -> DataDict:
        """ select the `idx`-th data, get rid of padded zeros and return it.

        Important data like x, y are all converted to ndarray.
        :param batch:
        :param idx:
        :return: DataDict
            Values can be an str or ndarray.
        """
        result = dict()
        for key, value in batch.items():
            if type(value) == str:
                result[key] = value
            elif type(value) == list:
                result[key] = value[idx]
            elif not key.startswith('T_'):
                T_xy = 'T_xs' if 'x' in key else 'T_ys'
                value = value[idx, :, :, :batch[T_xy][idx]]  # C, F, T
                value = value.permute(1, 2, 0).contiguous()  # F, T, C
                value = value.numpy()
                if value.shape[-1] == 2:
                    value = value.view(dtype=np.complex64)  # F, T, 1
                result[key] = value
        # if 'x_mag' in result:
        #     result['x'] = result['x_mag']
        #     del result['x_mag']
        return result

    @staticmethod
    def save_dirspec(fname: Union[str, Path], **kwargs):
        """ save directional spectrograms.

        :param fname:
        :param kwargs:
        :return:
        """
        scio.savemat(fname,
                     {hp.spec_data_names[k]: v
                      for k, v in kwargs.items() if k in hp.spec_data_names}
                     )

    def set_needs(self, **kwargs: Channel):
        """ set which data are needed.

        :param kwargs: available keywords are [x, y, x_phase, y_phase, speech_fname]
        """
        for k in self._needs:
            if k in kwargs:
                self._needs[k] = kwargs[k]

    # @xy_signature
    # def normalize(self, **kwargs):
    #     """

    #     :param x: input dirspec
    #     :param y: output dirspec
    #     :return:
    #     """

    #     for xy, v in kwargs.items():
    #         kwargs[xy] = None
    #         v_norm = LogModule.log(v)
    #         kwargs[xy] = self.norm_modules[xy].normalize_(v_norm)

    #     return kwargs

    # @xy_signature
    # def normalize_(self, **kwargs):
    #     """

    #     :param x: input dirspec
    #     :param y: output dirspec
    #     :return:
    #     """

    #     for xy, v in kwargs.items():
    #         v = LogModule.log_(v)
    #         kwargs[xy] = self.norm_modules[xy].normalize_(v)

    #     return kwargs

    # @xy_signature
    # def denormalize(self, **kwargs):
    #     """

    #     :param x: input dirspec
    #     :param y: output dirspec
    #     :return:
    #     """

    #     for xy, v in kwargs.items():
    #         kwargs[xy] = None
    #         v_denorm = self.norm_modules[xy].denormalize(v)
    #         kwargs[xy] = LogModule.exp_(v_denorm)

    #     return kwargs

    # @xy_signature
    # def denormalize_(self, **kwargs):
    #     """

    #     :param x: input dirspec
    #     :param y: output dirspec
    #     :return:
    #     """

    #     for xy, v in kwargs.items():
    #         v = self.norm_modules[xy].denormalize_(v)
    #         kwargs[xy] = LogModule.exp_(v)

    #     return kwargs

    # noinspection PyProtectedMember
    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: DirSpecDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[DirSpecDataset]
        """
        if type(dataset) != cls:
            raise TypeError
        n_split = len(ratio)
        ratio = [0, *ratio]
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[mask] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[mask] = 1 - ratio.sum()

        all_files = dataset._all_files
        ratio_cum = np.cumsum(ratio)
        i_boundaries = (ratio_cum * len(all_files)).astype(np.int)
        i_boundaries[-1] = len(all_files)

        dataset._all_files = None
        subsets = [copy(dataset) for _ in range(n_split)]
        for i, subset in enumerate(subsets):
            subset._all_files = all_files[i_boundaries[i]:i_boundaries[i+1]]
            subset._needs = subset._needs.copy()

        return subsets
