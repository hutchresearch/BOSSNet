import torch

class ConvolutionBlock1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvolutionBlock1D, self).__init__()
        self.conv_layer = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.elu = torch.nn.ELU(inplace=True)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.bn(out)
        out = self.elu(out)
        out = self.pool(out)
        return out