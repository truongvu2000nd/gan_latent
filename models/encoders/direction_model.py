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
        if batchnorm:
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
            in_features=size * 2,
            middle_features=size,
            out_features=size,
            bias=bias,
            batchnorm=batchnorm,
            final_norm=final_norm,
        )
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.alpha.data.fill_(1.0)

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
            final_norm=final_norm,
        )

    def forward(self, w1, w2):
        # return mask
        bs, _, _ = w1.size()
        return torch.sigmoid(self.net(torch.cat((w1, w2), dim=1).view(bs, -1))).view(
            bs, self.n_latent, self.size
        )


class MaskModelLSTM(nn.Module):
    def __init__(
        self,
        input_size=512,
        hidden_size=512,
        num_layers=2,
        dropout=0,
        bidirectional=False,
    ):
        """
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # self.fc = nn.Linear()

    def forward(self, w1: torch.Tensor, w2_plus: torch.Tensor):
        """
        Input:
            w1: N, 512 -- hidden
            w2: N, 14, 512 -- input sequence
        Output:
            mask: N, 14, 512 [binary or (0, 1)]
            w1.repeat(1, 14, 1) * mask + w2 * (1-mask)
        """
        h0 = w1.unsqueeze(0).repeat(self.num_layers * self.D, 1, 1)    # num_layers, N, 512
        c0 = torch.zeros(self.num_layers * self.D, w1.size(0), self.hidden_size, device=w1.device)

        out, _ = self.lstm(w2_plus, (h0, c0))
        return torch.sigmoid(out)

    def forward2(self, w1_plus, w2_plus):
        """
        Input:
            w1: N, 14, 512 
            w2: N, 14, 512 
            w=cat(w1, w2) [N, 14, 1024] -- input sequence with random hidden
        Output:
            mask: N, 14, 512 [binary or (0, 1)] -- return sequence
            w1 * mask + w2 * (1-mask)
        """
        bs = w1_plus.size(0)
        w_cat = torch.cat((w1_plus, w2_plus), dim=2)  # [N, 14, 1024]
        h0 = torch.zeros(self.num_layers * self.D, bs, self.hidden_size, device=w1_plus.device)
        c0 = torch.zeros(self.num_layers * self.D, bs, self.hidden_size, device=w1_plus.device)

        out, _ = self.lstm(w_cat, (h0, c0))
        return torch.sigmoid(out)


class MaskModelGRU(nn.Module):
    def __init__(self) -> None:
        """
            net: GRU 14
        """
        super().__init__()
        self.net = nn.GRU()

        self.fc = nn.Linear()

    def forward1(self, w1, w2_plus):
        """
        Input:
            w1: N, 512 -- hidden
            w2: N, 14, 512 -- input sequence
        Output:
            mask: N, 14, 512 [binary or (0, 1)]
            w1.repeat(1, 14, 1) * mask + w2 * (1-mask)
        """
        pass

    def forward2(self, w1_plus, w2_plus):
        """
        Input:
            w1: N, 14, 512 
            w2: N, 14, 512 
            w=cat(w1, w2) [N, 14, 1024] -- input sequence with random hidden
        Output:
            mask: N, 14, 512 [binary or (0, 1)] -- return sequence
            w1 * mask + w2 * (1-mask)
        """
        pass


if __name__ == "__main__":
    model = MaskModelLSTM(num_layers=2, input_size=1024, bidirectional=False)
    print(model)
    from torchinfo import summary

    w1 = torch.rand(1, 14, 512)
    w2 = torch.rand(1, 14, 512)

    inp = {"w1_plus": w1, "w2_plus": w2}
    summary(model, input_data=inp)
