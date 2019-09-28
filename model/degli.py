import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stft

from .istft import istft


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(7, 3), padding=None):
        super().__init__()
        if not padding:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, :ch//2, ...] * self.sigmoid(x[:, ch//2:, ...])
        return x


class DeGLI_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convglu_first = ConvGLU(6, 16, kernel_size=(11, 11))
        self.two_convglus = nn.Sequential(
            ConvGLU(16, 16),
            ConvGLU(16, 16)
        )
        self.convglu_last = ConvGLU(16, 16)
        self.conv = nn.Conv2d(16, 2, kernel_size=(7, 3), padding=(3, 1))

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
    return x / (torch.sqrt(x[:, :1]**2+x[:, 1:]**2) + 1e-8) * mag


class DeGLI(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, depth=1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.depth = depth
        self.window = nn.Parameter(data=torch.hann_window(n_fft), requires_grad=False)
        
        self.dnn = DeGLI_DNN()

    def forward(self, x, mag, max_length=None):
        for _ in range(self.depth):
            mag_replaced = replace_magnitude(x, mag)  # B, 2, F, T
            consistent = istft(
                mag_replaced.permute(0, 2, 3, 1), 
                max_length, 
                hop_length=self.hop_length,
            )  # B, F, T, 2

            consistent = stft(
                consistent,
                n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
            ).permute(0, 3, 1, 2).contiguous()  # B, 2, F, T
            residual = self.dnn(x, mag_replaced, consistent)
            x = consistent - residual
        out = replace_magnitude(x, mag)
        return x, out, residual
