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

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        n_samples = len(out)
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)
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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None, normalize=True):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        if normalize:
            features = F.normalize(features, dim=2)

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        print(mask)
        print(logits_mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos.view(anchor_count, batch_size).mean()

        return loss


class SupConLossWithMB(nn.Module):
    def __init__(self, temperature=0.1, dim=512, bank_size=512, init_queue=None):
        super(SupConLossWithMB, self).__init__()
        self.temperature = temperature

        self.dim = dim
        self.bank_size = bank_size
        if init_queue is not None:
            self.register_buffer("queue", init_queue)
        else:
            self.register_buffer("queue", torch.randn(self.bank_size, self.dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, feature1, feature_dir, nrep=0, normalize=True):
        device = torch.device("cuda") if feature1.is_cuda else torch.device("cpu")

        if normalize:
            feature1 = F.normalize(feature1, dim=1)
            # feature2 = F.normalize(feature2, dim=1)
            feature_dir = F.normalize(feature_dir, dim=1)

        batch_size = feature1.size(0)
        l_pos = torch.matmul(feature_dir, feature1.T)
        l_neg = torch.matmul(feature_dir, self.queue.T.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        mask_index = torch.cat([
            torch.arange(
                i * batch_size, i * batch_size + batch_size
            ).unsqueeze(1)
            for i in range(nrep + 1)
        ], dim=1).to(device)
        
        mask = torch.scatter(
            torch.zeros_like(logits), 1, mask_index, 1
        )

        log_prob = F.log_softmax(logits, dim=1)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = -mean_log_prob_pos.mean()

        with torch.no_grad():
            self.queue[batch_size:] = self.queue[:-batch_size].clone()
            self.queue[:batch_size] = feature_dir

        return loss


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
        x = (((imgs + 1) * 0.5).permute(0, 2, 3, 1) * 255).long()
        batch_boxes, _ = self.mtcnn.detect(x)
        outs, outs_bg = [], []
        for img, box in zip(imgs, batch_boxes):
            box = box[0].astype("int")

            img_bg = img.clone()
            img_bg[:, box[1] : box[3], box[0] : box[2]].zero_()
            outs_bg.append(img_bg)

            img = img[:, box[1] : box[3], box[0] : box[2]]
            out = F.interpolate(img.unsqueeze(0), size=(112, 112), mode="area")
            outs.append(out)
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


class BatchMTCNN(nn.Module):
    def __init__(self, device="cpu"):
        super(BatchMTCNN, self).__init__()
        self.mtcnn = MTCNN(image_size=112, device=device, select_largest=False)
        self.device = device

    def forward(self, imgs, img_size=112, return_masks_bg=True):
        x = (((imgs + 1) * 0.5).permute(0, 2, 3, 1) * 255).long()
        batch_boxes, _ = self.mtcnn.detect(x)
        outs = []
        masks_bg = []
        for img, box in zip(imgs, batch_boxes):
            box = box[0].astype("int")

            out = img[:, box[1] : box[3], box[0] : box[2]]
            out = F.interpolate(
                out.unsqueeze(0), size=(img_size, img_size), mode="area"
            )
            outs.append(out)

            mask_bg = torch.ones((img.size(2), img.size(3))).to(self.device)
            mask_bg[box[1] : box[3], box[0] : box[2]].zero_()
            masks_bg.append(mask_bg.unsqueeze(0).unsqueeze(0))

        outs = torch.cat(outs, dim=0)
        masks_bg = torch.cat(masks_bg, dim=0)
        
        if return_masks_bg:
            return outs, masks_bg
        return outs


if __name__ == "__main__":
    loss = SupConLossWithMB(bank_size=10, dim=4)
    x1 = torch.randn(4, 4)
    x2 = torch.randn(4, 4)
    x3 = torch.randn(4, 4)
    # x = F.normalize(x, dim=2)
    loss1, loss2 = loss(x1, x1, nrep=1)
    print(loss1 / loss2)
    loss1, loss2 = loss(x2, x3, nrep=1)
    print(loss1 / loss2)

    # loss = SupConLoss()

    # x1 = torch.randn(4, 2, 5)
    # loss(x1)
