import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import Backbone
from configs.paths_config import model_paths


class ContrastiveLoss(nn.Module):
    """Computes the contrastive loss

    Args:
        - k: the number of transformations per batch
        - temperature: temp to scale before exponential

    Shape:
        - Input: the raw, feature scores.
                tensor of size :math:`(k x minibatch, F)`, with F the number of features
                expects first axis to be ordered by transformations first (i.e., the
                first "minibatch" elements is for first transformations)
        - Output: scalar
    """

    def __init__(self, k: int, temp: float, abs: bool, reduce: str) -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.abs = abs
        self.reduce = reduce
        #         self.iter = 0

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        n_samples = len(out)
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        #         if (self.iter % 100) == 0:
        #             print(sim)
        #          self.iter += 1

        sim = torch.exp(sim * self.temp)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(self.k):
            start, end = i * (n_samples // self.k), (i + 1) * (n_samples // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, y):
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        return 1 - F.cosine_similarity(x_feats, y_feats)
