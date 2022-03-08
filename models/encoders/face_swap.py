import math
import torch.utils.model_zoo as model_zoo

import torch
from torch import nn
from torch.nn import functional as F

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, omega_only=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.omega_only = omega_only
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if not omega_only:
            self.layer5 = self._make_layer(block, 512, layers[4], stride=2,
                                        dilate=replace_stride_with_dilation[2])                 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c1 = self.layer2(x)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2)
        if not self.omega_only:
            c4 = self.layer5(c3)
            return c1, c2, c3, c4
        return c1, c2, c3

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet50(pretrained=False, omega_only=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3, 2], omega_only=omega_only)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
  return model

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

class FPN(nn.Module):
    def __init__(self, backbone, depth, omega_only):
        super(FPN, self).__init__()
        scale = 4 if depth==50 else 1
        self.omega_only = omega_only
        self.backbone = backbone
        self.output1 = conv_bn(128*scale, 512, stride=1)
        self.output2 = conv_bn(256*scale, 512, stride=1)
        self.output3 = conv_bn(512*scale, 512, stride=1)
        if not omega_only:
            self.output4 = nn.Sequential(
                                        nn.Conv2d(512*scale, 512, 3, 1, padding=1, bias=True),
                                        nn.BatchNorm2d(512)
                                        )

    def forward(self, x):
        if not self.omega_only:
            c1, c2, c3, c4 = self.backbone(x)
            p4 = self.output4(c4)                                               # N, 512, 4, 4
        else:
            c1, c2, c3 = self.backbone(x)
            p4 = None
        p3 = self.output3(c3)                                                   # N, 512, 8, 8
        p2 = self.output2(c2) + F.upsample(p3, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 16, 16
        p1 = self.output1(c1) + F.upsample(p2, scale_factor=2, mode='bilinear', align_corners=True) # N, 512, 32, 32

        return p4, p3, p2, p1

class styleMapping(nn.Module):
    def __init__(self, size):
        super(styleMapping, self).__init__()
        num_layers = int(math.log(size, 2))
        convs = []
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            convs.append(nn.Conv2d(512, 512, 3, 2, padding=1, bias=True))
            convs.append(nn.BatchNorm2d(512))
            convs.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*convs)
    def forward(self, x):
        x = self.convs(x).squeeze(2).squeeze(2).unsqueeze(1)
        return x

class HieRFE(nn.Module):
    def __init__(self, backbone, num_latents=[3, 4, 7], depth=50, omega_only=False):
        super(HieRFE, self).__init__()
        self.fpn = FPN(backbone, depth, omega_only)
        self.act = nn.Tanh()
        self.mapping1 = nn.ModuleList()
        for i in range(num_latents[0]):
            self.mapping1.append(styleMapping(8))
        self.mapping2 = nn.ModuleList()
        for i in range(num_latents[1]):
            self.mapping2.append(styleMapping(16))
        self.mapping3 = nn.ModuleList()
        for i in range(num_latents[2]):
            self.mapping3.append(styleMapping(32))
        
    def forward(self, x):
        latents = []
        f4, f8, f16, f32 = self.fpn(x)
        for maps in self.mapping1:
            latents.append(maps(f8))
        for maps in self.mapping2:
            latents.append(maps(f16))
        for maps in self.mapping3:
            latents.append(maps(f32))
        latents = torch.cat(latents, 1)
        return self.act(latents), f4
## ====================swapper================================================================================================================

class TransferCell(nn.Module):
    def __init__(self, num_blocks):
        super(TransferCell, self).__init__()
        self.num_blocks = num_blocks
        self.idd_selectors = nn.ModuleList()
        self.idd_shifters = nn.ModuleList()
        self.att_selectors = nn.ModuleList()
        self.att_shifters = nn.ModuleList()

        self.act = nn.LeakyReLU(True)

        for i in range(self.num_blocks):
            self.idd_selectors.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Sigmoid()))
            self.idd_shifters.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Tanh()))

            self.att_selectors.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Sigmoid()))
            self.att_shifters.append(nn.Sequential(nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 512), nn.Tanh()))


    def forward(self, idd, att):
        for i in range(self.num_blocks):
            fuse = torch.cat([idd, att], dim=1)
            idd = self.act(idd * self.idd_selectors[i](fuse) + self.idd_shifters[i](fuse))
            att = self.act(att * self.att_selectors[i](fuse) + self.att_shifters[i](fuse))
        return idd.unsqueeze(1), att.unsqueeze(1)

class InjectionBlock(nn.Module):
    def __init__(self,):
        super(InjectionBlock, self).__init__()
        self.idd_linears = nn.Sequential(nn.Linear(512, 512),nn.ReLU(True))
        self.idd_selectors = nn.Linear(512, 512)
        self.idd_shifters = nn.Linear(512, 512)
        self.att_bns = nn.BatchNorm1d(512, affine=False)

    def forward(self, x):
        idd, att = x[0], x[1]
        normalized = self.att_bns(att)
        actv = self.idd_linears(idd)
        gamma = self.idd_selectors(actv)
        beta = self.idd_shifters(actv)
        out = normalized * (1 + gamma) + beta
        return out

        
class InjectionResBlock(nn.Module):
    def __init__(self, num_blocks):
        super(InjectionResBlock, self).__init__()
        self.num_blocks = num_blocks

        self.att_path1 = nn.ModuleList()
        self.att_path2 = nn.ModuleList()

        self.act = nn.LeakyReLU(True)

        for i in range(self.num_blocks):
            self.att_path1.append(nn.Sequential(InjectionBlock(), nn.LeakyReLU(True), nn.Linear(512, 512)))
            self.att_path2.append(nn.Sequential(InjectionBlock(), nn.LeakyReLU(True), nn.Linear(512, 512)))


    def forward(self, idd, att):
        for i in range(self.num_blocks):
            att_bias = att*1
            att = self.att_path1[i]((idd, att))
            att = self.att_path2[i]((idd, att))
            att = att + att_bias
        return self.act(att.unsqueeze(1))


def LCR(idd, att, swap_indice=4):
    swapped = torch.cat([att[:, :swap_indice], idd[:, swap_indice:]], 1)
    return swapped


class FaceTransferModule(nn.Module):
    def __init__(self, num_blocks=1, swap_indice=4, num_latents=14, typ="ftm"):
        super(FaceTransferModule, self).__init__()
        self.type = typ
        if self.type == "ftm":
            self.swap_indice = swap_indice
            self.num_latents = num_latents - swap_indice
            self.blocks = nn.ModuleList()
            for i in range(self.num_latents):
                self.blocks.append(TransferCell(num_blocks))

            self.weight = nn.Parameter(torch.randn(1, self.num_latents, 512))
        
        elif self.type == "injection":
            self.swap_indice = swap_indice
            self.num_latents = num_latents - swap_indice
            self.blocks = nn.ModuleList()
            for i in range(self.num_latents):
                self.blocks.append(InjectionResBlock(num_blocks))
        
        elif self.type == "lcr":
            self.swap_indice = swap_indice
        
        else:
            raise NotImplementedError()
        
    def forward(self, idd, att):
        if self.type == "ftm":
            att_low = att[:, :self.swap_indice]
            idd_high = idd[:, self.swap_indice:]
            att_high = att[:, self.swap_indice:]

            N = idd.size(0)
            idds = []
            atts = []
            for i in range(self.num_latents):
                new_idd, new_att = self.blocks[i](idd_high[:, i], att_high[:, i])
                idds.append(new_idd)
                atts.append(new_att)
            idds = torch.cat(idds, 1)
            atts = torch.cat(atts, 1)
            scale = torch.sigmoid(self.weight).expand(N, -1, -1)
            latents = scale * idds + (1-scale) * atts

            return torch.cat([att_low, latents], 1)
            
        elif self.type == "injection":
            att_low = att[:, :self.swap_indice]
            idd_high = idd[:, self.swap_indice:]
            att_high = att[:, self.swap_indice:]

            N = idd.size(0)
            latents = []
            for i in range(self.num_latents):
                new_latent = self.blocks[i](idd_high[:, i], att_high[:, i])
                latents.append(new_latent)
            latents = torch.cat(latents, 1)
            return torch.cat([att_low, latents], 1)
        
        elif self.type == "lcr":
            return LCR(idd, att, swap_indice=self.swap_indice)
        
        else:
            raise NotImplementedError()