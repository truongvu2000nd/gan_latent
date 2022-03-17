import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Union, List


def create_mlp(
    depth: int,
    in_features: int,
    middle_features: int,
    out_features: int,
    bias: bool = True,
    batchnorm: bool = True,
    momentum: float = 0.1,
    final_norm: bool = False,
):
    # initial dense layer
    layers = []
    layers.append(
        (
            "linear_1",
            torch.nn.Linear(
                in_features, out_features if depth == 1 else middle_features
            ),
        )
    )

    #  iteratively construct batchnorm + relu + dense
    for i in range(depth - 1):
        if batchnorm:
            layers.append(
                (f"batchnorm_{i+1}", torch.nn.BatchNorm1d(num_features=middle_features, momentum=momentum))
            )
        layers.append((f"relu_{i+1}", torch.nn.ReLU()))
        layers.append(
            (
                f"linear_{i+2}",
                torch.nn.Linear(
                    middle_features,
                    out_features if i == depth - 2 else middle_features,
                    False if i == depth - 2 else bias,
                ),
            )
        )

    if final_norm:
        layers.append(
            (f"batchnorm_{depth}", torch.nn.BatchNorm1d(num_features=out_features, momentum=momentum))
        )

    # return network
    return torch.nn.Sequential(OrderedDict(layers))


class FixedMaskModel(nn.Module):
    def __init__(self, size=512):
        super().__init__()
        self.size = size
        self.mask = nn.Parameter(torch.zeros(size), requires_grad=True)
        nn.init.uniform_(self.mask)

    def forward(self, w1, w2):
        return w1 * self.mask + w2 * (1 - self.mask)


class MaskModel(nn.Module):
    def __init__(self, size=512, depth=3, bias=True, batchnorm=False, final_norm=False):
        super().__init__()
        self.size = size
        self.net = create_mlp(
            depth=depth,
            in_features=size * 2,
            middle_features=size,
            out_features=size,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=final_norm,
        )

    def forward(self, w1, w2):
        return torch.sigmoid(self.net(torch.cat((w1, w2), dim=1)))


class WPlusMaskModel(nn.Module):
    def __init__(
        self,
        size=512,
        n_latent=18,
        depth=3,
        bias=True,
        batchnorm=False,
        momentum=0.1,
        final_norm=False,
    ):
        super().__init__()
        self.size = size
        self.n_latent = n_latent

        self.net = create_mlp(
            depth=depth,
            in_features=size * n_latent * 2,
            middle_features=size,
            out_features=size * n_latent,
            bias=bias,
            batchnorm=batchnorm,
            momentum=momentum,
            final_norm=final_norm,
        )

    def forward(self, w1, w2):
        # return mask
        bs, _, _ = w1.size()
        return torch.sigmoid(self.net(torch.cat((w1, w2), dim=1).view(bs, -1))).view(
            bs, self.n_latent, self.size
        )


class Highway(nn.Module):
    def __init__(self, size, num_layers, act="relu", momentum=0.1):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, size),
                nn.BatchNorm1d(size, momentum=momentum)
            ) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "lrelu":
            self.act = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.act(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x


class MaskHighway(nn.Module):
    def __init__(self, size=256, n_latent=18, num_layers=5, act="relu"):
        super().__init__()
        self.fc1s = nn.ModuleList([nn.Linear(1024, size) for _ in range(n_latent)])
        self.fc2s = nn.ModuleList([nn.Linear(size, 512) for _ in range(n_latent)])
        self.highways = nn.ModuleList([Highway(size, num_layers, act=act) for _ in range(n_latent)])

    def forward(self, w1, w2):
        w_cat = torch.cat((w1, w2), dim=2)
        mask = []
        for i in range(w_cat.size(1)):
            out = self.fc1s[i](w_cat[:, i])
            out = self.highways[i](out)
            out = self.fc2s[i](out)
            mask.append(out.unsqueeze(1))
        mask = torch.cat(mask, dim=1)
        return torch.sigmoid(mask)


if __name__ == "__main__":
    from torchinfo import summary

    w1, w2 = torch.randn(1, 18, 512), torch.randn(1, 18, 512)
    model = MaskHighway(num_layers=3, act="lrelu")
    summary(model, input_size=[(1,18,512), (1,18,512)])
