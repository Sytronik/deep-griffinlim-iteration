from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypeVar, Union

import numpy as np
import scipy.io as scio
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from hparams import Channel, hp

TensArr = Union[ndarray, Tensor]
StrOrSeq = TypeVar('StrOrSeq', str, Sequence[str])
TupOrSeq = TypeVar('TupOrSeq', tuple, Sequence[tuple])
DataDict = Dict[str, Any]


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
                    if data.dtype.type == np.str_:
                        sample[k] = str(data.item())
                    elif data.dtype == np.int:
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
