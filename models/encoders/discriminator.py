from torch import nn


class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim=512, n_mlp=4, get_first_codes=7):
        super().__init__()

        self.style_dim = style_dim
        self.get_first_codes = get_first_codes

        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w, ):
        if w.ndim == 3 and self.get_first_codes:
            w = w[:, :self.get_first_codes, :].reshape(-1, 512)
        elif w.ndim == 3:
            w = w.view(-1, 512)
        return self.mlp(w)


if __name__ == '__main__':
    from torchinfo import summary

    model = LatentCodesDiscriminator()
    summary(model, (1, 18, 512), get_first_codes=None)