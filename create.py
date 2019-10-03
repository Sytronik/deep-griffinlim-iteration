import multiprocessing as mp
import os
from argparse import ArgumentParser

import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray
from tqdm import tqdm

from hparams import hp


def save_feature(i_speech: int, s_path_speech: str, speech: ndarray) -> tuple:
    spec_clean = np.ascontiguousarray(librosa.stft(speech, **hp.kwargs_stft))
    mag_clean = np.ascontiguousarray(np.abs(spec_clean)[..., np.newaxis])
    signal_power = np.mean(np.abs(speech)**2)
    list_dict = []
    list_snr_db = []
    for _ in enumerate(args.num_snr):
        snr_db = -6*np.random.rand()
        list_snr_db.append(snr_db)
        snr = librosa.db_to_power(snr_db)
        noise_power = signal_power / snr
        noisy = speech + np.sqrt(noise_power) * np.random.randn(len(speech))
        spec_noisy = librosa.stft(noisy, **hp.kwargs_stft)
        spec_noisy = np.ascontiguousarray(spec_noisy)

        list_dict.append(
            dict(spec_noisy=spec_noisy,
                 spec_clean=spec_clean,
                 mag_clean=mag_clean,
                 path_speech=s_path_speech,
                 length=len(speech),
                 )
        )
    return i_speech, list_snr_db, list_dict


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('kind_data', choices=('TRAIN', 'TEST'))
    args = hp.parse_argument(parser, print_argument=False)
    args.kind_data = args.kind_data.lower()

    path_speech_folder = hp.dict_path[f'speech_{args.kind_data}']
    flist_speech = (list(path_speech_folder.glob('**/*.WAV')) +
                    list(path_speech_folder.glob('**/*.wav')))

    path_feature = hp.dict_path[f'feature_{args.kind_data}']
    os.makedirs(path_feature, exist_ok=True)

    pool = mp.Pool(
        processes=mp.cpu_count()//2 - 1,
        initializer=lambda: np.random.seed(os.getpid()),
    )
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
        i_speech, list_snr_db, list_dict = result.get()
        for snr_db, dict_result in zip(list_snr_db, list_dict):
            np.savez(path_feature / hp.form_feature.format(i_speech, snr_db),
                     **dict_result,
                     )
