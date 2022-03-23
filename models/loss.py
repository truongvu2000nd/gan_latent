import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import Backbone, IR_101
from configs.paths_config import model_paths
from facenet_pytorch import MTCNN, InceptionResnetV1


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


class ArcFaceLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(ArcFaceLoss, self).__init__()
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
        self.facenet.eval()
        self.mtcnn = MTCNN(image_size=112, device=device)

    def extract_imgs(self, imgs):
        x = (((imgs+1)*0.5).permute(0, 2, 3, 1) * 255).long()
        batch_boxes , _ = self.mtcnn.detect(x)
        outs, outs_bg = [], []
        for img, box in zip(imgs, batch_boxes):
            box = box[0].astype("int")
            img = img[:, box[1]:box[3], box[0]:box[2]]
            img_bg = img.clone()
            img_bg[:, box[1]:box[3], box[0]:box[2]].zero_()
            out = F.interpolate(img.unsqueeze(0), size=(112, 112), mode="area")
            outs.append(out)
            outs_bg.append(img_bg)
        outs = torch.cat(outs, dim=0)
        outs_bg = torch.cat(outs_bg, dim=0)
        return outs, outs_bg

    def extract_feats(self, imgs):
        x_feats = self.facenet(imgs)
        return x_feats

    def forward(self, x, y):
        x, _ = self.extract_imgs(x)
        y, _ = self.extract_imgs(y)
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        return F.cosine_similarity(x_feats, y_feats)


class CircularFaceLoss(nn.Module):
    def __init__(self, device="cpu"):
        super(CircularFaceLoss, self).__init__()
        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(model_paths["circular_face"]))
        self.facenet.eval()
        self.mtcnn = MTCNN(image_size=112, device=device)

    def extract_imgs(self, imgs):
        x = (((imgs+1)*0.5).permute(0, 2, 3, 1) * 255).long()
        batch_boxes , _ = self.mtcnn.detect(x)
        outs, outs_bg = [], []
        for img, box in zip(imgs, batch_boxes):
            box = box[0].astype("int")
            img = img[:, box[1]:box[3], box[0]:box[2]]
            img_bg = img.clone()
            img_bg[:, box[1]:box[3], box[0]:box[2]].zero_()
            out = F.interpolate(img.unsqueeze(0), size=(112, 112), mode="area")
            outs.append(out)
            outs_bg.append(img_bg)
        outs = torch.cat(outs, dim=0)
        outs_bg = torch.cat(outs_bg, dim=0)
        return outs, outs_bg

    def extract_feats(self, imgs):
        x_feats = self.facenet(imgs)
        return x_feats

    def forward(self, x, y):
        x, _ = self.extract_imgs(x)
        y, _ = self.extract_imgs(y)
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        return F.cosine_similarity(x_feats, y_feats)
