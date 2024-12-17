"""
The BOSS Net model.

This file contains the Model object.

MIT License

Copyright (c) 2024 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch 

from bossnet.model.modules.conv_block_1d import ConvolutionBlock1D
from bossnet.model.modules.res_block_1d import ResNetBlock1D
from bossnet.model.modules.linear_block import LinearBlock
from bossnet.model.modules.positional_encodings import PositionalEncoding1D

def exists(obj):
    if obj is not None:
        return True
    return False

class BossNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        model_dim,
        dim_mults,
        kernel_size,
        num_targets,
        adapt_pool_output_dim,
        num_linear,
        linear_dim,
        pos_enc,
    ) -> None:
        super(BossNet, self).__init__()

        padding = kernel_size // 2

        self.pos_enc = PositionalEncoding1D(1) if pos_enc else None

        self.first_conv = ConvolutionBlock1D(
            in_channels=1 if not pos_enc else 2,
            out_channels=dim_mults[0] * model_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.res_blocks = torch.nn.ModuleList()

        dims = [*map(lambda m: model_dim * m, dim_mults)]
        in_out_channels = list(zip(dims[:-1], dims[1:]))

        for in_channels, out_channels in in_out_channels:
            res_block = ResNetBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.res_blocks.append(res_block)

        self.adapt_pool = torch.nn.AdaptiveAvgPool1d(output_size=adapt_pool_output_dim)

        self.linear_input_size = out_channels * adapt_pool_output_dim

        self.linear_sequential = torch.nn.Sequential(
            *[
                LinearBlock(self.linear_input_size, linear_dim),
                *[
                    LinearBlock(linear_dim, linear_dim)
                    for _ in range(num_linear - 1)
                ],
                torch.nn.Linear(linear_dim, num_targets),
            ]
        )

    def forward(self, x):
        if exists(self.pos_enc):
            pos = self.pos_enc(x)
            x = torch.hstack([x, pos])
        out = self.first_conv(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.adapt_pool(out)
        linear_input = out.view(-1, self.linear_input_size)
        linear_output = self.linear_sequential(linear_input)
        return linear_output
