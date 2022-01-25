import torch
from torch import nn
from torchvision import models


def _build_backbone_inception(net: nn.Module, depth: int) -> nn.Module:
    new_backbone = nn.Sequential()
    for child in net.named_children():
        if child[0] == "Mixed_6a":
            break
        else:
            new_backbone.add_module(str(child[0]), child[1])
    return new_backbone



class DeepEndovo(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone_t = _build_backbone_inception(models.inception_v3(pretrained=True), 3)
        self.backbone_t1 = _build_backbone_inception(models.inception_v3(pretrained=True), 3)
        self.lstm = nn.LSTM(input_size=484416, hidden_size=1000, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(
            in_features=1000,
            out_features=6
        )

    def forward(self, x):
        feature_outputs = []
        # images == [b, T, C, H, W]
        for t in range(1, x.shape[1]):
            ot = self.backbone_t(x[:, t, :, :, :])
            ot1 = self.backbone_t1(x[:, t - 1, :, :, :])
            output = torch.cat([ot, ot1], dim=1)
            feature_outputs.append(output.view(output.shape[0], -1))
        feature_outputs = torch.stack(feature_outputs).permute(1, 0, 2)
        # [b, t, 64]
        output, (hn, cn) = self.lstm(feature_outputs)
        # out == [b, t, 32]
        output = self.classifier(output)
        # we do not include (N, t,Hout​)=0
        return output