"""
Forked from https://github.com/seungwonpark/istft-pytorch/blob/master/istft_deconv.py
"""
import torch
import torch.nn.functional as F


def istft(stft_matrix, length, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True):
    """stft_matrix = (batch, freq, time, 2))
    - Based on librosa implementation and Keunwoo Choi's implementation
        - librosa: http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
        - Keunwoo Choi's: https://gist.github.com/keunwoochoi/2f349e72cc941f6f10d4adf9b0d3f37e#file-istft-torch-py
    """
    assert normalized == False
    assert onesided == True
    assert window == 'hann'
    assert center == True

    device = stft_matrix.device
    n_batch = stft_matrix.shape[0]
    n_fft = 2 * (stft_matrix.shape[1] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device)

    n_frames = stft_matrix.shape[2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    conj = torch.tensor([1., -1.], requires_grad=False, device=device)

    # [a,b,c,d,e] -> [a,b,c,d,e,d,c,b]
    stft_matrix = torch.cat(
        (stft_matrix, conj*stft_matrix.flip(dims=(1,))[:, 1:-1]), dim=1)
    # now shape is [n_batch, n_fft, T, 2]

    stft_matrix = stft_matrix.transpose(1, 2)
    stft_matrix = torch.ifft(stft_matrix, signal_ndim=2)[..., 0]  # get real part of ifft
    ytmp = stft_matrix * istft_window
    ytmp = ytmp.transpose(1, 2)
    # ytmp = ytmp.unsqueeze(0)
    # now [n_batch, n_fft, T]. this is stack of `ytmp` in librosa/core/spectrum.py

    eye = torch.eye(n_fft, requires_grad=False, device=device)
    eye = eye.unsqueeze(1)  # [n_fft, 1, n_fft]

    y = F.conv_transpose1d(ytmp, eye, stride=hop_length, padding=0)
    y = y.view(n_batch, -1)
    assert y.size(-1) == expected_signal_len

    y = y[:, n_fft//2:]
    y = y[:, :length] if length else y[:, :-n_fft//2]

    # this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    coeff = n_fft/float(hop_length) / 2.0
    return y / coeff
