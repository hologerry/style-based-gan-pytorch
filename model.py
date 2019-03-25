import random

import torch
from torch import nn
from torch.nn import functional as F

from network import (ConvBlock, EqualConv2d, EqualLinear, PixelNorm,
                     StyledConvBlock)


class Generator(nn.Module):
    def __init__(self, code_dim):
        super().__init__()

        self.progression = nn.ModuleList([StyledConvBlock(512, 512, 3, 1, initial=True),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 512, 3, 1),
                                          StyledConvBlock(512, 256, 3, 1),
                                          StyledConvBlock(256, 128, 3, 1),
                                          StyledConvBlock(128,  64, 3, 1)])

        self.to_rgb = nn.ModuleList([EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(512, 3, 1),
                                     EqualConv2d(256, 3, 1),
                                     EqualConv2d(128, 3, 1),
                                     EqualConv2d(64,  3, 1)])

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                upsample = F.interpolate(
                    out, scale_factor=2, mode='bilinear', align_corners=False)
                # upsample = self.blur(upsample)
                out = conv(upsample, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out
                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, noise=None, step=0, alpha=-1, mean_style=None, style_weight=0, mixing_range=(-1, -1)):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size,
                                         size, device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(
                    mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1),
                                          ConvBlock(256, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(513, 512, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, 128, 1),
                                       EqualConv2d(3, 256, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1),
                                       EqualConv2d(3, 512, 1)])

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5,
                                    mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(
                        input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
