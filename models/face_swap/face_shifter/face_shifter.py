from turtle import forward
from .AEI_Net import AEI_Net
from .backbone import Backbone

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceShifter(nn.Module):
    def __init__(self, face_shifter_path=None, arcface_path=None):
        super().__init__()

        self.G = AEI_Net(c_id=512)
        self.G.eval()
        self.G.load_state_dict(torch.load(face_shifter_path))

        self.arcface = Backbone(50, 0.6, 'ir_se')
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load(arcface_path), strict=False)

    def forward(self, x_src, x_tgt):
        embeds = self.arcface(F.interpolate(x_src[:, :, 19:237, 19:237],
                             (112, 112), mode='bilinear', align_corners=True))
        
        out, _ = self.G(x_tgt, embeds)
        return out
