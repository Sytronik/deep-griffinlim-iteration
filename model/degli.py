import torch
import torch.nn as nn

from .istft import InverseSTFT


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(7, 7), padding=None, batchnorm=False):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        if batchnorm:
            self.conv = nn.Sequential(
                self.conv,
                nn.BatchNorm2d(out_ch * 2)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, :ch//2, ...] * self.sigmoid(x[:, ch//2:, ...])
        return x


class DeGLI_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convglu_first = ConvGLU(6, 16, kernel_size=(11, 11), batchnorm=True)
        self.two_convglus = nn.Sequential(
            ConvGLU(16, 16, batchnorm=True),
            ConvGLU(16, 16)
        )
        self.convglu_last = ConvGLU(16, 16)
        self.conv = nn.Conv2d(16, 2, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x, mag_replaced, consistent):
        x = torch.cat([x, mag_replaced, consistent], dim=1)
        x = self.convglu_first(x)
        residual = x
        x = self.two_convglus(x)
        x += residual
        x = self.convglu_last(x)
        x = self.conv(x)
        return x


def replace_magnitude(x, mag):
    phase = torch.atan2(x[:, 1:], x[:, :1])  # imag, real
    return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


class DeGLI(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, depth=1, separate_dnns=True, out_all_block=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.depth = depth
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.istft = InverseSTFT(n_fft, hop_length=self.hop_length, window=self.window.data)
        self.dnns = nn.ModuleList([DeGLI_DNN() for _ in range(depth if separate_dnns else 1)])
        self.out_all_block = out_all_block

    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)

    def forward(self, x, mag, max_length=None, depth=0):
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        x_list = [x]
        for i in range(depth if depth > 0 else self.depth):
            # B, 2, F, T
            mag_replaced = replace_magnitude(x, mag)

            # B, F, T, 2
            inv = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_length)
            consistent = self.stft(inv)

            # B, 2, F, T
            consistent = consistent.permute(0, 3, 1, 2).contiguous()
            residual = self.dnns[min(i, len(self.dnns)-1)](x, mag_replaced, consistent)
            x = consistent - residual
            x_list.append(x)
        out = replace_magnitude(x, mag)
        return x_list if self.out_all_block else x_list[-1:], out, residual
