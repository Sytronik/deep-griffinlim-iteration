from itertools import zip_longest

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
        ch_hidden = 16
        self.convglu_first = ConvGLU(6, ch_hidden, kernel_size=(11, 11), batchnorm=True)
        self.two_convglus = nn.Sequential(
            ConvGLU(ch_hidden, ch_hidden, batchnorm=True),
            ConvGLU(ch_hidden, ch_hidden)
        )
        self.convglu_last = ConvGLU(ch_hidden, ch_hidden)
        self.conv = nn.Conv2d(ch_hidden, 2, kernel_size=(7, 7), padding=(3, 3))

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
    def __init__(self, n_fft: int, hop_length: int,
                 depth=1, out_all_block=False, final_avg=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out_all_block = out_all_block
        self.final_avg = final_avg

        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.istft = InverseSTFT(n_fft, hop_length=self.hop_length, window=self.window.data)

        self.dnns = nn.ModuleList([DeGLI_DNN() for _ in range(depth)])

    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)

    def forward(self, x, mag, max_length=None, repeat=1):
        if isinstance(max_length, torch.Tensor):
            max_length = max_length.item()

        out_blocks = []
        x_repeat = x
        for ii in range(repeat):
            in_blocks = [x_repeat]
            for dnn in self.dnns:
                # B, 2, F, T
                mag_replaced = replace_magnitude(in_blocks[-1], mag)

                # B, F, T, 2
                waves = self.istft(mag_replaced.permute(0, 2, 3, 1), length=max_length)
                consistent = self.stft(waves)

                # B, 2, F, T
                consistent = consistent.permute(0, 3, 1, 2)
                residual = dnn(in_blocks[-1], mag_replaced, consistent)
                in_blocks.append(consistent - residual)
            if self.final_avg:
                x_repeat = sum(in_blocks[1:]) / (len(in_blocks) - 1)
                if self.out_all_block or ii == repeat - 1:
                    out_blocks.append(x_repeat)
                continue
            x_repeat = in_blocks[-1]
            if self.out_all_block:
                out_blocks.append(x_repeat)
            elif ii == repeat - 1:
                out_blocks = in_blocks[-1:]

        # out_blocks = in_blocks[1:] if self.out_all_block else in_blocks[-1:]
        out_blocks = torch.stack(out_blocks, dim=1)

        final_out = replace_magnitude(x_repeat, mag)
        # final_out = replace_magnitude(out_blocks.mean(dim=1), mag)

        return out_blocks, final_out, residual
