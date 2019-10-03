"""
istft implementation using PyTorch

The function signature convention is the same as `torch.stft`.
The implementation is based on `librosa.istft`.

"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def overlap_add(x, hop_length, eye=None):
    """
    x: B, W, T
    eye: identity matrix of size (W, W)
    return: B, W + hop_length * (T - 1)
    """
    n_batch, W, _ = x.shape
    if eye is None:
        eye = torch.eye(W, device=x.device)

    x = F.conv_transpose1d(x, eye, stride=hop_length, padding=0)
    x = x.view(n_batch, -1)
    return x


class InverseSTFT(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, window=None):
        super().__init__()

        self.n_fft = n_fft

        # Set the default hop if it's not already specified
        if hop_length is None:
            self.hop_length = int(win_length // 4)
        else:
            self.hop_length = hop_length

        if win_length is None:
            win_length = n_fft

        # kernel for overlap_add
        eye = torch.eye(n_fft)
        self.eye = nn.Parameter(eye.unsqueeze(1), requires_grad=False)  # n_fft, 1, n_fft

        # default window is a rectangular window (convention of torch.stft)
        if window is None:
            window = torch.ones(win_length)
        else:
            assert win_length == len(window)

        # pad window so that its length is n_fft
        diff = n_fft - win_length
        window = F.pad(window.unsqueeze(0), [diff//2, math.ceil(diff/2)])
        window.unsqueeze_(2)  # 1, n_fft, 1

        # square of window for calculating the numerical error occured during stft & istft
        self.win_sq = nn.Parameter(window**2, requires_grad=False)  # 1, n_fft, 1
        self.win_sq_sum = None

        # ifft basis * window
        # The reason why this basis is used instead of torch.ifft is
        # because torch.ifft / torch.irfft randomly cause segfault
        # when the model is in nn.DataParallel
        # of PyTorch 1.2.0 (py3.7_cuda10.0.130_cudnn7.6.2_01.2)
        eye_realimag = torch.stack((eye, torch.zeros(n_fft, n_fft)), dim=-1)
        basis = torch.ifft(eye_realimag, signal_ndim=1)  # n_fft, n_fft, 2
        basis[..., 1] *= -1  # because (a+b*1j)*(c+d*1j) == a*c - b*d
        basis *= window
        self.basis = nn.Parameter(basis, requires_grad=False)  # n_fft, n_fft, 2

    def forward(self, stft_matrix,
                center=True, normalized=False, onesided=True, length=None):
        """stft_matrix: (n_batch (B), n_freq, n_frames (T), 2))
        if `onesided == True`, `n_freq == n_fft` should be satisfied.
        else, `n_freq == n_fft // 2+ 1` should be satisfied.

        """
        n_batch, n_freq, n_frames, _ = stft_matrix.shape

        assert ((not onesided) and (n_freq == self.n_fft)
                or onesided and (n_freq == self.n_fft // 2 + 1))

        if length:
            padded_length = length
            if center:
                padded_length += self.n_fft
            n_frames = min(n_frames, math.ceil(padded_length / self.hop_length))

        stft_matrix = stft_matrix[:, :, :n_frames]

        if onesided:
            flipped = stft_matrix[:, 1:-1].flip(1)
            flipped[..., 1] *= -1
            stft_matrix = torch.cat((stft_matrix, flipped), dim=1)
            # now stft_matrix is (B, n_fft, T, 2)

        # The reason why this basis is used instead of torch.ifft is
        # because torch.ifft / torch.irfft randomly cause segfault
        # when the model is in nn.DataParallel
        # of PyTorch 1.2.0 (py3.7_cuda10.0.130_cudnn7.6.2_01.2)
        ytmp = torch.einsum('bftc,fwc->bwt', stft_matrix, self.basis)
        y = overlap_add(ytmp, self.hop_length, self.eye)
        # now y is (B, n_fft + hop_length * (n_frames - 1))

        # compensate numerical errors of window function
        if self.win_sq_sum is None or self.win_sq_sum.shape[1] != y.shape[1]:
            win_sq = self.win_sq.expand(1, -1, n_frames)  # 1, n_fft, n_frames
            win_sq_sum = overlap_add(win_sq, self.hop_length, self.eye)
            win_sq_sum[win_sq_sum <= torch.finfo(torch.float32).tiny] = 1.
            # now win_sq_sum is (1, y.shape[1])
            self.win_sq_sum = win_sq_sum

        y /= self.win_sq_sum

        if center:
            y = y[:, self.n_fft // 2:]
        if length is not None:
            if length < y.shape[1]:
                y = y[:, :length]
            else:
                y = F.pad(y, [0, length - y.shape[1]])
            # now y is (B, length)

        if normalized:
            y *= self.n_fft**0.5

        return y


def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, normalized=False, onesided=True, length=None):
    """stft_matrix: (n_batch (B), n_freq, n_frames (T), 2))
    if `onesided == True`, `n_freq == n_fft` should be satisfied.
    else, either `n_freq == n_fft // 2` or `n_freq == n_fft // 2+ 1` should be satisfied.

    """
    if onesided:
        n_fft = 2 * (stft_matrix.shape[1] - 1)
    else:
        n_fft = stft_matrix.shape[1]
    module = InverseSTFT(n_fft, hop_length, win_length, window).to(device=stft_matrix.device)
    y = module(stft_matrix, center, normalized, onesided, length)
    return y


if __name__ == '__main__':
    import librosa

    n_fft = 2048
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length)

    # stft & istft using torch
    original = torch.randn(1, 10000)
    stft_mat = torch.stft(
        original, n_fft,
        hop_length=hop_length, win_length=win_length, window=window,
    )
    recon = istft(
        stft_mat,
        hop_length=hop_length, win_length=win_length, window=window, length=10000,
    )

    # stft & istft using librosa
    original_np = original.numpy().squeeze()
    stft_mat_np = librosa.stft(
        original_np, n_fft,
        hop_length=hop_length, win_length=win_length, window='hann',
    )
    recon_np = librosa.istft(
        stft_mat_np,
        hop_length=hop_length, win_length=win_length, window='hann', length=10000,
    )
    rms_error = ((recon_np-recon.squeeze().numpy())**2).mean()**0.5
    print(rms_error)
