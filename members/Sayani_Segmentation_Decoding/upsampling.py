
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):

    def __init__(self, mode="bilinear"):
        super(Upsample, self).__init__()

        self.mode = mode

    def forward(self, x, target_size):

        x = F.interpolate(
            x,
            size=target_size,
            mode=self.mode,
            align_corners=False if self.mode == "bilinear" else None
        )

        return x
