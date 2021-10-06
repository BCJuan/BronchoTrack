import torch
from torch import nn
from torchvision import models


class OffsetNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = models.resnet34(pretrained=True)
        self.backbone_t1 = models.resnet34(pretrained=True)
        self.classifier = nn.Linear(
            in_features=2000,
            out_features=6
        )

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.backbone_t(x[:, t, :, :, :])
            ot1 = self.backbone_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            feature_outputs.append(self.classifier(output))   
        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        return feature_outputs
