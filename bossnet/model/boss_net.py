from functools import partial
import torch 

from model.modules.conv_block_1d import ConvolutionBlock1D
from model.modules.res_block_1d import ResNetBlock1D
from model.modules.linear_block import LinearBlock
from model.modules.positional_encodings import PositionalEncoding1D

class BossNet(torch.nn.Module):
    def __init__(self) -> None:
        super(BossNet, self).__init__()

        self.metadatanet = (
            torch.nn.Sequential(
                torch.nn.Linear(14, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 1024),
            )
        )

        self.pos_enc = PositionalEncoding1D(1)

        self.first_conv = ConvolutionBlock1D(
            in_channels=2,
            out_channels=4,
            kernel_size=30,
            padding=15,
        )

        res_block = partial(ResNetBlock1D,
            kernel_size=30,
            padding=15,
            metadata_emb_dim=1024,
        )

        self.res_blocks = torch.nn.ModuleList([
            res_block(
                in_channels=4,
                out_channels=8,
            ),
            res_block(
                in_channels=8,
                out_channels=16,
            ),
            res_block(
                in_channels=16,
                out_channels=32,
            ),
            res_block(
                in_channels=32,
                out_channels=64,
            ),
        ])

        self.adapt_pool = torch.nn.AdaptiveAvgPool1d(output_size=1024)

        self.linear_sequential = torch.nn.Sequential(
            LinearBlock(64 * 1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            torch.nn.Linear(1024, 3),
        )

    def forward(self, x, m):
        meta_out = self.metadatanet(m)
        pos = self.pos_enc(x)
        x = torch.hstack([x, pos])
        out = self.first_conv(x)
        for res_block in self.res_blocks:
            out = res_block(out, meta_out)
        out = self.adapt_pool(out)
        linear_input = out.view(-1, 64 * 1024)
        linear_output = self.linear_sequential(linear_input)
        return linear_output

