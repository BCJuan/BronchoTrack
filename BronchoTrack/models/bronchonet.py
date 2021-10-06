import torch
from torch import nn
from torchvision import models
from .modules import InvertedResidual


class BronchoNetSingleTemporal(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=64)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(x.shape[1]):
            output = x[:, t, :, :, :]
            feature_outputs.append(self.backbone(output))
        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 64]
        output, (hn, cn) = self.lstm(feature_outputs)
        # out == [b, t, 32]
        output = self.linear(output)
        # we do not include (N, t,Hout​)=0
        return output[:, 1:, :]


class BronchoNetDoubleTemporalEarlyFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_t = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        self.conv_t1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.features[0] = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=64)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.conv_t(x[:, t, :, :, :])
            ot1 = self.conv_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            feature_outputs.append(self.backbone(output))   
        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 64]
        output, (hn, cn) = self.lstm(feature_outputs)
        # out == [b, t, 32]
        output = self.linear(output)
        # we do not include (N, t,Hout​)=0
        return output


class BronchoNetDoubleTemporalLateFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = self._build_backbone(models.efficientnet_b0(pretrained=True), 5)
        self.backbone_t1 = self._build_backbone(models.efficientnet_b0(pretrained=True), 5)

        self.fusion = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.latebone = nn.Sequential(
            InvertedResidual(64, 64, 1),
            InvertedResidual(64, 32, 2),
            InvertedResidual(32, 32, 1),
            InvertedResidual(32, 16, 2)
        )
        # NOTE: i know it is 356 for depth 5 of backbone, but we could adapt
        # automatically to any depth by making half forward and introudcing
        # shape in the model
        self.linear1 = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.SiLU(inplace=True))
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = nn.Linear(32, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.backbone_t(x[:, t, :, :, :])
            ot1 = self.backbone_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            o = self.fusion(output)
            o = self.latebone(o)
            o = self.linear1(o.view(o.shape[0], -1))
            feature_outputs.append(o)

        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 64]
        output, (hn, cn) = self.lstm(feature_outputs)
        # out == [b, t, 32]
        output = self.linear(output)
        # we do not include (N, t,Hout​)=0
        return output

    def _build_backbone(self, net: nn.Module, depth: int) -> nn.Module:
        new_backbone = nn.Sequential()
        for child in net.named_children():
            if child[0] == "features":
                for j, grandchild in enumerate(child[1][:depth]):
                    new_backbone.add_module(str(j), grandchild)
        return new_backbone


class BronchoNetDoubleLateFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = self._build_backbone(models.efficientnet_b0(pretrained=True), 5)
        self.backbone_t1 = self._build_backbone(models.efficientnet_b0(pretrained=True), 5)

        self.fusion = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.latebone = nn.Sequential(
            InvertedResidual(64, 64, 1),
            InvertedResidual(64, 32, 2),
            InvertedResidual(32, 32, 1),
            InvertedResidual(32, 16, 2)
        )
        # NOTE: i know it is 356 for depth 5 of backbone, but we could adapt
        # automatically to any depth by making half forward and introudcing
        # shape in the model
        self.linear1 = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 6))

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.backbone_t(x[:, t, :, :, :])
            ot1 = self.backbone_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            o = self.fusion(output)
            o = self.latebone(o)
            o = self.linear1(o.view(o.shape[0], -1))
            feature_outputs.append(o)

        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 6]
        return feature_outputs

    def _build_backbone(self, net: nn.Module, depth: int) -> nn.Module:
        new_backbone = nn.Sequential()
        for child in net.named_children():
            if child[0] == "features":
                for j, grandchild in enumerate(child[1][:depth]):
                    new_backbone.add_module(str(j), grandchild)
        return new_backbone
