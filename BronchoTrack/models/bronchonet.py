import torch
from torch import nn
from torchvision import models


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

        """How to extract efficientnet backbone until layer n
        for child in self.backbone.named_childred():
            if child[0] == "features":
                for grandchild in child[1]:
                    <here append children to list and make sequential>
        """

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
        return output[:, 1:, :]


# TODO: do late fusion double temporal