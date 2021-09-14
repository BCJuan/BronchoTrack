import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torch._C import dtype
from .models.bronchonet import BronchoNet


class BronchoModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = BronchoNet()
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self._shared_step(batch, batch_idx)
        return test_loss

    def _shared_step(self, batch, batch_idx):
        x, py, ry = batch["images"], batch["pos_labels"], batch["rot_labels"]
        z = self.model(x)
        loss = self.loss(z[:, :, :3], py) + self.loss(z[:, :, 3:], ry)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer