import torch

class LinearBlock(torch.nn.Module):
    def __init__(self, input_dim, linear_dim):
        super(LinearBlock, self).__init__()
        self.linear_layer = torch.nn.Linear(input_dim, linear_dim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear_layer(x)
        out = self.relu(out)
        return out