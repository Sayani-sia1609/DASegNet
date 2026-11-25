
import torch
import torch.nn as nn


class SkipConnection(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):

        super(SkipConnection, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):

        # Combine decoder features with encoder features
        x = torch.cat([x, skip], dim=1)

        # Refine the combined features
        x = self.conv(x)

        return x
