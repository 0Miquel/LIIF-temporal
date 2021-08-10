import torch.nn as nn

from models import register


@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        self.block1 = nn.Linear(in_dim, hidden_list[0])

        self.block2 = ResBlockMLP(hidden_list[:3])
        self.block3 = ResBlockMLP(hidden_list[1:])

        block4 = []
        block4.append(nn.ReLU())
        block4.append(nn.Linear(hidden_list[-1], out_dim))
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.block1(x.view(-1, x.shape[-1]))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(*shape, -1)


class ResBlockMLP(nn.Module):
    def __init__(self, dims):
        super(ResBlockMLP, self).__init__()
        m = []

        m.append(nn.ReLU())
        m.append(nn.Linear(dims[0], dims[1]))
        m.append(nn.ReLU())
        m.append(nn.Linear(dims[1], dims[2]))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res