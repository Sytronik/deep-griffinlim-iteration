from typing import Sequence, Dict

import librosa
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from hparams import hp
from matlab_lib import Evaluation as EvalModule

EVAL_METRICS = EvalModule.metrics


def calc_snrseg(y_clean: ndarray, y_est: ndarray, T_ys: Sequence[int] = (0,)) \
        -> float:
    """ calculate snrseg. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if len(T_ys) == 1 and y_clean.shape[0] != 1:
        if T_ys == (0,):
            T_ys = (y_clean.shape[0],)
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]

    sum_result = np.float32(0.)
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        # T
        norm_clean = np.einsum(
            'ftc,ftc->t', item_clean[:, :T, :], item_clean[:, :T, :]
        )
        err = item_est[:, :T, :] - item_clean[:, :T, :]
        norm_err = np.einsum(
            'ftc,ftc->t', err, err
        ) + np.finfo(np.float32).eps

        snrseg = np.log10(norm_clean / norm_err + np.finfo(np.float32).eps)
        np.minimum(snrseg, _LIM_UPPER, out=snrseg)
        np.maximum(snrseg, _LIM_LOWER, out=snrseg)
        sum_result += snrseg.mean()
    sum_result *= 10

    return sum_result


def calc_using_eval_module(y_clean: ndarray, y_est: ndarray,
                           T_ys: Sequence[int] = (0,)) -> Dict[str, float]:
    """ calculate metric using EvalModule. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    if len(T_ys) > 1:
        metrics = None
        sum_result = None
        for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
            # noinspection PyArgumentList
            metrics, result = EvalModule(item_clean[:T], item_est[:T], hp.fs)
            result = np.array(result)
            if sum_result is None:
                sum_result = result
            else:
                sum_result += result
        sum_result = sum_result.tolist()
    else:
        # noinspection PyArgumentList
        metrics, sum_result = EvalModule(y_clean[0, :T_ys[0]], y_est[0, :T_ys[0]], hp.fs)

    return {k: v for k, v in zip(metrics, sum_result)}


def reconstruct_wave(*args: ndarray, n_iter=0, n_sample=-1) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag_spectrogram, phase_spectrogram) or (complex_spectrogram,)
    :param n_iter: no. of iteration of griffin-lim. 0 for not using griffin-lim.
    :param n_sample: number of samples of output wave
    :return:
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.istft(mag * np.exp(1j * phase), **hp.kwargs_istft)

        phase = np.angle(librosa.stft(wave, **hp.kwargs_stft))

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.istft(spec, **hp.kwargs_istft, **kwarg_len)

    return wave


def draw_spectrogram(data: ndarray, to_db=True, show=False, dpi=150, **kwargs):
    """
    
    :param data:
    :param to_db:
    :param show:
    :param dpi:
    :param kwargs: vmin, vmax
    :return: 
    """

    if to_db:
        # data[data == 0] = data[data > 0].min()
        data = librosa.amplitude_to_db(data)
    data = data.squeeze()

    fig, ax = plt.subplots(dpi=dpi,)
    ax.imshow(data,
              cmap=plt.get_cmap('CMRmap'),
              extent=(0, data.shape[1], 0, hp.fs // 2),
              origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(ax.images[0])
    if show:
        fig.show()

    return fig
