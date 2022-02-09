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
        layers.append(
            (f"batchnorm_{i+1}", torch.nn.BatchNorm1d(num_features=middle_features))
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
            (f"batchnorm_{depth}", torch.nn.BatchNorm1d(num_features=out_features))
        )

    # return network
    return torch.nn.Sequential(OrderedDict(layers))


class SingleDirectionModel(torch.nn.Module):
    """K directions nonlinearly conditional on latent code"""

    def __init__(
        self,
        size: int,
        depth: int,
        alpha: Union[float, List[float]] = 0.1,
        normalize: bool = True,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        # self.alpha = alpha
        self.normalize = normalize

        # make mlp net
        self.net = create_mlp(
            depth=depth,
            in_features=size*2,
            middle_features=size,
            out_features=size,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=final_norm,
        )
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.alpha.data.fill_(1.)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z = torch.cat((z1, z2), dim=1)
        z1_norm = z1.norm(dim=1).view(-1, 1)
        dz = self.net(z)
        if self.normalize:
            dz = F.normalize(dz, dim=1)
        return dz * z1_norm * self.alpha

    def sample_alpha(self) -> float:
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            return self.alpha
        return np.random.uniform(self.alpha[0], self.alpha[1], size=1)[0]


class DirectionModel(torch.nn.Module):
    """K directions nonlinearly conditional on latent code"""

    def __init__(
        self,
        k: int,
        size: int,
        depth: int,
        alpha: Union[float, List[float]] = 0.1,
        normalize: bool = True,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.k = k
        self.size = size
        self.alpha = alpha
        self.normalize = normalize

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for _ in range(k):
            net = create_mlp(
                depth=depth,
                in_features=size,
                middle_features=size,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=final_norm,
            )
            self.nets.append(net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z = z.repeat((self.k, 1, 1,))

        #  calculate directions
        dz = []
        for i in range(self.k):
            res_dz = self.nets[i](z[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)

        #  add directions
        z = z + dz

        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))

    def sample_alpha(self) -> float:
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            return self.alpha
        return np.random.uniform(self.alpha[0], self.alpha[1], size=1)[0]

    def post_process(self, dz: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            norm = torch.norm(dz, dim=1)
            dz = dz / torch.reshape(norm, (-1, 1))
        return self.sample_alpha() * dz


class FixedMaskModel(nn.Module):
    def __init__(self, size=512):
        super().__init__()
        self.size = size
        self.mask = nn.Parameter(torch.zeros(size), requires_grad=True)
        nn.init.uniform_(self.mask)

    def forward(self, w1, w2):
        return w1 * self.mask + w2 * (1 - self.mask)


class MaskModel(nn.Module):
    def __init__(self, size=512, depth=3, bias=True, batchnorm=True, final_norm=False):
        super().__init__()
        self.size = size
        self.net = create_mlp(
            depth=depth,
            in_features=size,
            middle_features=size,
            out_features=size,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=final_norm,
        )


    def get_mask(self, w1, w2):
        return F.sigmoid(self.net(torch.cat((w1, w2), dim=1)))

    def forward(self, w1, w2):
        mask = self.get_mask(w1, w2)
        return w1 * mask + w2 * (1 - mask)


if __name__ == "__main__":
    model = DirectionModel(k=5, size=512, depth=3)
    print(model)
