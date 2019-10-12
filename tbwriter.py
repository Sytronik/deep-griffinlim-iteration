import multiprocessing as mp
from typing import Dict

import librosa
import numpy as np
import torch
from numpy import ndarray
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from hparams import hp
from utils import (EVAL_METRICS,
                   calc_using_eval_module,
                   draw_spectrogram,
                   reconstruct_wave,
                   )


class CustomWriter(SummaryWriter):
    def __init__(self, *args, group='', **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        if group == 'train':
            dict_custom_scalars = dict(loss=['Multiline', ['loss/train',
                                                           'loss/valid']])
        else:
            dict_custom_scalars = dict()

        for i, m in enumerate(EVAL_METRICS):
            dict_custom_scalars[f'{i+1}_{m}'] = [
                'Multiline', [f'{group}/{i+1}_{m}/GLim',
                              f'{group}/{i+1}_{m}/DeGLI']
            ]

        self.add_custom_scalars({group: dict_custom_scalars})

        self.pool_eval_module = mp.pool.ThreadPool(1)

        # x, y
        self.reused_sample = dict()
        self.dict_eval_glim = None

        # fig
        self.kwargs_fig = dict()

        # audio
        self.y_scale = 1.

    def write_one(self, step: int,
                  out: ndarray = None,
                  res: ndarray = None,
                  suffix: str = None,
                  **kwargs: ndarray) -> ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param step:
        :param out:
        :param eval_with_y_ph: determine if `out` is evaluated with `y_phase`
        :param kwargs: keywords can be [x, y, x_phase, y_phase]

        :return: evaluation result
        """

        assert out is not None and res is not None
        result_eval_glim = self.write_x_y(kwargs, step) if kwargs else None

        assert self.reused_sample
        y_wav = self.reused_sample['y_wav']
        length = self.reused_sample['length']
        out = out.squeeze()

        out_wav = reconstruct_wave(out, n_sample=length)

        result_eval = self.pool_eval_module.apply_async(
            calc_using_eval_module,
            (y_wav, out_wav)
        )
        # dict_eval = calc_using_eval_module(
        #     y_wav,
        #     out_wav_y_ph if eval_with_y_ph else out_wav
        # )

        if hp.draw_test_fig or self.group == 'train':
            fig_out = draw_spectrogram(np.abs(res), **self.kwargs_fig)
            self.add_figure(f'{self.group}/4_DNN_Output_Spectrum{suffix}', fig_out, step)

        self.add_audio(f'{self.group}/3_DeGLI_output',
                       torch.from_numpy(out_wav / self.y_scale),
                       step,
                       sample_rate=hp.fs)

        if result_eval_glim:
            self.dict_eval_glim = result_eval_glim.get()
        dict_eval = result_eval.get()
        for i, m in enumerate(dict_eval.keys()):
            j = i + 1
            self.add_scalar(f'{self.group}/{j}_{m}/GLim', self.dict_eval_glim[m], step)
            self.add_scalar(f'{self.group}/{j}_{m}/DeGLI{suffix}', dict_eval[m], step)

        return np.array([list(dict_eval.values()),
                         list(self.dict_eval_glim.values())], dtype=np.float32)

    def write_x_y(self, kwargs: Dict[str, ndarray], step: int) -> mp.pool.AsyncResult:
        # F, T, 1
        x = kwargs['x'].squeeze()
        y_mag = kwargs['y_mag'].squeeze()
        y = kwargs['y'].squeeze()
        length = kwargs['length']

        # T,
        x_wav = reconstruct_wave(x, n_sample=length)
        glim_wav = reconstruct_wave(y_mag, np.angle(x), n_iter=hp.n_glim_iter, n_sample=length)
        glim_mag = np.abs(librosa.stft(glim_wav, **hp.kwargs_stft))
        y_wav = reconstruct_wave(y, n_sample=length)
        self.y_scale = np.abs(y_wav).max() / 0.5
        result_eval_glim = self.pool_eval_module.apply_async(
            calc_using_eval_module,
            (y_wav, glim_wav)
        )
        # result_eval_glim = None
        # self.dict_eval_glim = calc_using_eval_module(y_wav, glim_wav[:y_wav.shape[0]])

        if hp.draw_test_fig or self.group == 'train':
            ymin = y_mag[y_mag > 0].min()
            vmin, vmax = librosa.amplitude_to_db(np.array((ymin, y_mag.max())))
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)

            fig_x = draw_spectrogram(np.abs(x))
            fig_glim = draw_spectrogram(glim_mag, **self.kwargs_fig)
            fig_y = draw_spectrogram(y_mag)
            fig_glimres = draw_spectrogram(glim_mag-y_mag, **self.kwargs_fig)

            self.add_figure(f'{self.group}/0_Noisy_Spectrum', fig_x, step)
            self.add_figure(f'{self.group}/1_GLim_Spectrum', fig_glim, step)
            self.add_figure(f'{self.group}/2_Clean_Spectrum', fig_y, step)
            self.add_figure(f'{self.group}/3_Residual_Spectrum', fig_glimres, step)

        self.add_audio(f'{self.group}/0_Noisy_Wave',
                       torch.from_numpy(x_wav / (np.abs(x_wav).max() / 0.5)),
                       step,
                       sample_rate=hp.fs)
        self.add_audio(f'{self.group}/1_Clean_Wave',
                       torch.from_numpy(y_wav / self.y_scale),
                       step,
                       sample_rate=hp.fs)
        self.add_audio(f'{self.group}/2_GLim_Wave',
                       torch.from_numpy(glim_wav / self.y_scale),
                       step,
                       sample_rate=hp.fs)
        self.reused_sample = dict(x=x,
                                  y_wav=y_wav,
                                  length=length,
                                  )
        return result_eval_glim
