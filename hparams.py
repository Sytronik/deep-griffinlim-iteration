import os
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray


# noinspection PyArgumentList
class Channel(Enum):
    ALL = slice(None)
    LAST = slice(-1, None)
    NONE = None


@dataclass
class _HyperParameters:
    # devices
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 2
    num_workers: int = 4

    # select dataset
    # feature: str = 'IV'
    # feature: str = 'DirAC'
    # feature: str = 'mulspec'
    # room_train: str = 'room1+2+3'
    # room_test: str = 'room1+2+3'
    # room_create: str = ''

    # model_name: str = 'UNet'

    # feature parameters
    fs: int = 16000
    n_fft: int = 512
    l_frame: int = 512
    n_freq: int = 257
    l_hop: int = 256
    num_snr: int = 3

    # training
    n_data: int = 0  # <=0 to use all data
    train_ratio: float = 0.70
    n_epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 5e-4
    thr_clip_grad: float = 4.
    weight_decay: float = 1e-3  # Adam weight_decay

    # summary
    period_save_state: int = 5
    draw_test_fig: bool = False
    n_save_block_outs: int = 0
    n_glim_iter: int = 100
    repeat_train: int = 2
    repeat_test: int = 32

    # paths
    # logdir will be converted to type Path in the init_dependent_vars function
    logdir: str = f'./result/test'
    path_speech: Path = Path('./data/TIMIT')
    path_feature: Path = Path('./data')
    # path_feature: Path = Path('./backup')
    sfx_featuredir: str = ''

    # file names
    form_feature: str = '{:04d}_{:+.2f}dB.npz'  # i_speech, snr_db
    form_result: str = 'spec_{}.mat'

    # defined in __post_init__
    channels: Dict[str, Channel] = field(init=False)
    model: Dict[str, Any] = field(init=False)
    scheduler: Dict[str, Any] = field(init=False)
    spec_data_names: Dict[str, str] = field(init=False)

    # dependent variables
    dummy_input_size: Tuple = None
    dict_path: Dict[str, Path] = None
    kwargs_stft: Dict[str, Any] = None
    kwargs_istft: Dict[str, Any] = None

    def __post_init__(self):
        self.channels = dict(path_speech=Channel.NONE,
                             x=Channel.ALL,
                             y=Channel.ALL,
                             y_mag=Channel.ALL,
                             length=Channel.ALL,
                             )

        self.model = dict(n_fft=self.n_fft,
                          hop_length=self.l_hop,
                          depth=2,
                          out_all_block=True,
                          )
        self.scheduler = dict(mode='min',
                              factor=0.6,
                              patience=5,
                              verbose=False,
                              threshold=0.01,
                              threshold_mode='rel',
                              cooldown=0,
                              min_lr=1e-5,
                              eps=1e-08
                              )

        self.spec_data_names = dict(x='spec_noisy', y='spec_clean',
                                    y_mag='mag_clean',
                                    path_speech='path_speech',
                                    length='length',
                                    out='spec_estimated',
                                    res='spec_dnn_output',
                                    )

    def init_dependent_vars(self):
        self.logdir = Path(self.logdir)

        self.dummy_input_size = [
            (2,
             self.n_freq,
             int(2**np.floor(np.log2(4 / 3 * self.n_freq)))),
            (1,
             self.n_freq,
             int(2**np.floor(np.log2(4 / 3 * self.n_freq))))
        ]

        # path
        self.dict_path = dict(
            speech_train=self.path_speech / 'TRAIN',
            speech_test=self.path_speech / 'TEST',

            feature_train=self.path_feature / 'TRAIN',
            feature_test=self.path_feature / 'TEST',

            # normconst_train=path_feature_train / 'normconst.npz',

            figures=Path('./figures'),
        )

        # dirspec parameters
        self.kwargs_stft = dict(hop_length=self.l_hop, window='hann', center=True,
                                n_fft=self.n_fft, dtype=np.complex64)
        self.kwargs_istft = dict(hop_length=self.l_hop, window='hann', center=True,
                                 dtype=np.float32)

    @staticmethod
    def is_featurefile(f: os.DirEntry) -> bool:
        return (f.name.endswith('.npz')
                and not f.name.startswith('metadata')
                and not f.name.startswith('normconst'))

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True) -> Namespace:
        def set_attr_to_parsed(obj: Any, attr_name: str, attr_type: type, parsed: str):
            if parsed == '':
                return
            try:
                v = eval(parsed)
            except:
                v = None
            if attr_type == str or v is None or type(v) != attr_type:
                if (parsed.startswith("'") and parsed.endswith("'")
                        or parsed.startswith('"') and parsed.endswith('"')):
                    parsed = parsed[1:-1]
                if isinstance(obj, dict):
                    obj[attr_name] = parsed
                else:
                    setattr(obj, attr_name, parsed)
            else:
                if isinstance(obj, dict):
                    obj[attr_name] = v
                else:
                    setattr(obj, attr_name, v)

        if not parser:
            parser = ArgumentParser()
        args_already_added = [a.dest for a in parser._actions]
        dict_self = asdict(self)
        for k in dict_self:
            if hasattr(args_already_added, k):
                continue
            if isinstance(dict_self[k], dict):
                for sub_k in dict_self[k]:
                    parser.add_argument(f'--{k}--{sub_k}', default='')
            else:
                parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            if isinstance(dict_self[k], dict):
                for sub_k, sub_v in dict_self[k].items():
                    parsed = getattr(args, f'{k}__{sub_k}')
                    set_attr_to_parsed(getattr(self, k), sub_k, type(sub_v), parsed)
            else:
                parsed = getattr(args, k)
                set_attr_to_parsed(self, k, type(dict_self[k]), parsed)

        self.init_dependent_vars()
        if print_argument:
            print(repr(self))

        return args

    def __repr__(self):
        result = ('-------------------------\n'
                  'Hyper Parameter Settings\n'
                  '-------------------------\n')

        result += '\n'.join(
            [f'{k}: {v}' for k, v in asdict(self).items() if not isinstance(v, ndarray)])
        result += '\n-------------------------'
        return result


hp = _HyperParameters()
