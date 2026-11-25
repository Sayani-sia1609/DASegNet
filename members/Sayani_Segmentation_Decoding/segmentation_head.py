
import torch
import torch.nn as nn


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, num_classes):

        super(SegmentationHead, self).__init__()

        self.classifier = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(in_channels // 2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels // 2,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):

        # Generate raw segmentation logits
        logits = self.classifier(x)

        return logits
