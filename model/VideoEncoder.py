import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MPSConv3d(nn.Conv3d):
    """nn.Conv3d that runs R(2+1)D's factorized kernels -- spatial (1,k,k), temporal (k,1,1) and pointwise --
    as conv2d when its input is on an Apple-silicon GPU (MPS), where conv3d is slow. Same parameters and
    state_dict, same result (2e-15 max difference in float64); a training step of the video branch is ~2.6x
    faster on an M4 Max. On any other device this is a regular nn.Conv3d."""

    def forward(self, x):
        if (x.device.type != "mps" or self.groups != 1 or self.dilation != (1, 1, 1)
                or self.padding_mode != "zeros" or isinstance(self.padding, str)):
            return super().forward(x)
        (kt, kh, kw), (st, sh, sw), (pt, ph, pw) = self.kernel_size, self.stride, self.padding
        B, C, T, H, W = x.shape
        if kt == 1 and pt == 0: # spatial or pointwise kernel: fold time into the batch
            if st > 1:
                x = x[:, :, ::st]
                T = x.shape[2]
            y = F.conv2d(x.transpose(1, 2).reshape(B * T, C, H, W), self.weight[:, :, 0], self.bias, (sh, sw), (ph, pw))
            return y.reshape(B, T, *y.shape[1:]).transpose(1, 2)
        if kh == 1 and kw == 1 and ph == 0 and pw == 0: # temporal kernel: fold space into the width
            if sh > 1 or sw > 1:
                x = x[:, :, :, ::sh, ::sw]
                H, W = x.shape[3], x.shape[4]
            y = F.conv2d(x.reshape(B, C, T, H * W), self.weight[..., 0], self.bias, (st, 1), (pt, 0))
            return y.reshape(B, y.shape[1], y.shape[2], H, W)
        return super().forward(x)


def r2plus1d_18_kinetics():
    """torchvision's R(2+1)D-18 with its Kinetics-400 weights (what pretrained=True gave)."""
    try:
        from torchvision.models.video import R2Plus1D_18_Weights
        return models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    except ImportError: # torchvision < 0.13
        return models.video.r2plus1d_18(pretrained=True)


class VideoModel(nn.Module):
    def __init__(self, num_channels=3, model_name=None):

        super(VideoModel, self).__init__()
        if model_name is not None:
            model_name = 'x3d_s'
            self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
            self.model.blocks[5].activation = nn.Identity() # .ModuleList
            self.model.blocks[5].proj = nn.Linear(in_features=self.model.blocks[5].proj.in_features, out_features=768)# 512
        else:
            self.model = r2plus1d_18_kinetics()
            self.model.fc = nn.Sequential(nn.Dropout(0.1),
                                            nn.Linear(in_features=self.model.fc.in_features, out_features=768))# 512
            if num_channels == 4:
                new_first_layer = nn.Conv3d(in_channels=4,
                                            out_channels=self.model.stem[0].out_channels,
                                            kernel_size=self.model.stem[0].kernel_size,
                                            stride=self.model.stem[0].stride,
                                            padding=self.model.stem[0].padding,
                                            bias=False)
                # copy pre-trained weights for first 3 channels
                new_first_layer.weight.data[:, 0:3] = self.model.stem[0].weight.data
                self.model.stem[0] = new_first_layer
            for module in self.model.modules():
                if type(module) is nn.Conv3d:
                    module.__class__ = MPSConv3d
        self.modes = ["clip"]

    def forward(self, x):
        out = self.model(x)
        return out
