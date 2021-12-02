import torch
from torch import nn
from torchvision import models
from .modules import InvertedResidual


def _build_backbone(net: nn.Module, depth: int) -> nn.Module:
    new_backbone = nn.Sequential()
    for child in net.named_children():
        if child[0] == "features":
            for j, grandchild in enumerate(child[1][:depth]):
                new_backbone.add_module(str(j), grandchild)
    return new_backbone


class BronchoNetSingleTemporal(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = _build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.latebone = nn.Sequential(
            InvertedResidual(1280, 256, 2),
            InvertedResidual(256, 64, 2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(x.shape[1]):
            output = x[:, t, :, :, :]
            output = self.latebone(self.backbone(output))
            o = output.view(output.shape[0], -1)
            feature_outputs.append(o)
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
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.conv_t1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.backbone = _build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone[0] = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.latebone = nn.Sequential(
            InvertedResidual(1280, 256, 2),
            InvertedResidual(256, 64, 2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.conv_t(x[:, t, :, :, :])
            ot1 = self.conv_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            o = self.backbone(output)
            o = self.latebone(o)
            feature_outputs.append(o.view(o.shape[0], -1))
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
        self.backbone_t = _build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone_t1 = _build_backbone(models.efficientnet_b0(pretrained=True), 9)

        self.fusion = nn.Sequential(
            nn.Conv2d(2560, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        self.latebone = nn.Sequential(
            InvertedResidual(256, 128, 2),
            InvertedResidual(128, 64, 2)
        )
        # NOTE: i know it is 356 for depth 5 of backbone, but we could adapt
        # automatically to any depth by making half forward and introudcing
        # shape in the model
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 6)

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


class BronchoNetDoubleLateFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone_t1 = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)

        self.fusion = nn.Sequential(
            nn.Conv2d(2560, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.latebone = nn.Sequential(
            InvertedResidual(256, 128, 2),
            InvertedResidual(128, 64, 2)
        )
        # NOTE: i know it is 356 for depth 5 of backbone, but we could adapt
        # automatically to any depth by making half forward and introudcing
        # shape in the model
        self.linear1 = nn.Linear(256, 6)

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


class BronchoNetDoubleEarlyFusion(nn.Module):

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
            nn.Linear(in_features=1280, out_features=256)
        )
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
            ot = self.conv_t(x[:, t, :, :, :])
            ot1 = self.conv_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            feature_outputs.append(self.backbone(output))   
        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 64]
        output = self.linear1(output)
        # we do not include (N, t,Hout​)=0
        return output


class BronchoNet3DDoubleTemporal(nn.Module):

    def __init__(self):
        super().__init__()
        self.init3d = nn.Sequential(
            nn.Conv3d(3, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.conv3dblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(64, 64, 3, 1, 1),
                nn.BatchNorm3d(64),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv3d(64, 64, 3, 1, 1),
                nn.BatchNorm3d(64),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv3d(64, 64, 3, 1, 1),
                nn.BatchNorm3d(64),
                nn.ReLU())])
        self.globalpooling = nn.AdaptiveAvgPool3d((1, None, None))
        self.backbone_t = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone_t1 = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone_t[0] = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.backbone_t1[0] = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.latebone = nn.Sequential(
            InvertedResidual(1280, 256, 2)
        )

    def forward(self, x):
        # images == [b, T, C, H, W]
        feature_outputs = []
        x = x.permute(0, 2, 1,  3, 4)
        output = self.init3d(x)
        for module in self.conv3dblocks:
            output += module(output)
        for t in range(1, output.shape[2]):
            ot = self.backbone_t(output[:, :, t, :, :])
            ot1 = self.backbone_t1(output[:, :, t - 1, :, :])
            output = torch.cat([ot, ot1], dim=1)
            feature_outputs.append(self.latebone(output))  
        output = self.latebone(self.backbone(self.globalpooling(output).squeeze(2)))
        print(output.shape)
        return output


class BronchoNetDoubleLate3DFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)
        self.backbone_t1 = self._build_backbone(models.efficientnet_b0(pretrained=True), 9)

        self.fusion = nn.Sequential(
            nn.Conv2d(2560, 256, kernel_size=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3dblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 256, (3, 3, 3), 1, (1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv3d(256, 256, (3, 3, 3), 1, (1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU()),
            nn.Sequential(
                nn.Conv3d(256, 256, (3, 3, 3), 1, (1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU())])
        self.conv3dfinal = nn.Sequential(
                nn.Conv3d(256, 256, (3, 3, 3), 1, (1, 0, 0)),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((None, 1, 1)))
        # NOTE: i know it is 356 for depth 5 of backbone, but we could adapt
        # automatically to any depth by making half forward and introudcing
        # shape in the model
        self.linear1 = nn.Linear(256, 6)

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.backbone_t(x[:, t, :, :, :])
            ot1 = self.backbone_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            o = self.fusion(output)
            feature_outputs.append(o)
        feature_outputs = torch.stack(feature_outputs).permute(1, 2, 0, 3, 4)
        for module in self.conv3dblocks:
            feature_outputs = module(feature_outputs) + feature_outputs
        feature_outputs = self.conv3dfinal(feature_outputs).squeeze(3).squeeze(3).permute(0, 2, 1)
        feature_outputs = self.linear1(feature_outputs)
        return feature_outputs

    def _build_backbone(self, net: nn.Module, depth: int) -> nn.Module:
        new_backbone = nn.Sequential()
        for child in net.named_children():
            if child[0] == "features":
                for j, grandchild in enumerate(child[1][:depth]):
                    new_backbone.add_module(str(j), grandchild)
        return new_backbone