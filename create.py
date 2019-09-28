import os
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp

import numpy as np
import librosa
from tqdm import tqdm
from numpy import ndarray
import soundfile as sf

from hparams import hp


def save_feature(i_speech: int, s_path_speech: str, speech: ndarray):
    snr_db = -6*np.random.rand()
    snr = librosa.db_to_power(snr_db)
    signal_power = np.mean(np.abs(speech)**2)
    noise_power = signal_power / snr
    noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
    spec_clean = librosa.stft(speech, **hp.kwargs_stft)
    spec_noisy = librosa.stft(noisy, **hp.kwargs_stft)
    mag_clean = np.abs(spec_clean)
    dict_result = dict(spec_noisy=spec_noisy,
                       spec_clean=spec_clean,
                       mag_clean=mag_clean[..., np.newaxis],
                       path_speech=s_path_speech,
                       length=len(speech),
                       )
    return i_speech, snr_db, dict_result


def random_seed():
    np.random.seed(os.getpid())


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('kind_data', choices=('TRAIN', 'TEST'))

    args = hp.parse_argument(parser, print_argument=False)
    args.kind_data = args.kind_data.lower()
    path_speech_folder = hp.dict_path[f'speech_{args.kind_data}']
    path_feature = hp.dict_path[f'feature_{args.kind_data}']
    os.makedirs(path_feature, exist_ok=True)
    flist_speech = (list(path_speech_folder.glob('**/*.WAV')) +
                    list(path_speech_folder.glob('**/*.wav')))

    pool = mp.Pool(mp.cpu_count()//2 - 1, initializer=random_seed)
    results = []
    for i_speech, path_speech in enumerate(flist_speech):
        speech = sf.read(str(path_speech))[0].astype(np.float32)
        results.append(
            pool.apply_async(save_feature,
                             (i_speech, str(path_speech), speech),
                             )
        )

    pbar = tqdm(results, dynamic_ncols=True)
    for result in pbar:
        i_speech, snr_db, dict_result = result.get()
        np.savez(path_feature / hp.form_feature.format(i_speech, snr_db),
                 **dict_result,
                 )
