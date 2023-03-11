import torch

class ResNetBlock1D(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, metadata_emb_dim
    ):
        super(ResNetBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp = (
            torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(metadata_emb_dim, out_channels * 2),
            )
        )

        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.elu = torch.nn.ELU(inplace=True)
        padding_conv2 = padding - 1 if (kernel_size % 2 == 0) else padding
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding_conv2,
            bias=False,
        )

        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, m_emb):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.out_channels != self.in_channels:
            residual = residual.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            residual = torch.nn.functional.pad(residual, (ch1, ch2), "constant", 0)
            residual = residual.transpose(-1, -2)

        m_emb = self.mlp(m_emb)[:, :, None]
        scale, shift = torch.chunk(m_emb, 2, dim=1)
        residual = residual.clone() * (1 + scale) + shift

        out += residual
        out = self.elu(out)

        return out