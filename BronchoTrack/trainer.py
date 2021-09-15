import pathlib
import os
from pickle import load
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from .models.bronchonet import BronchoNetSingleTemporal
from .metrics import EuclideanDistance, NeedleError, DirectionError


class BronchoModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = BronchoNetSingleTemporal()
        self.loss = nn.MSELoss()
        parent_dir = pathlib.Path(__file__).parent.absolute()
        self.scaler = load(open(os.path.join(parent_dir, "data", "scaler.pkl"), "rb"))
        self.pos_error_train = EuclideanDistance()
        self.dir_error_train = DirectionError()
        self.ned_error_train = NeedleError()

    def training_step(self, batch, batch_idx):
        loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        return {"loss": loss, "preds": z, "targets": (py, ry)}

    def training_step_end(self, outputs):
        perror, derror = self._compute_errors(outputs)
        self.log("Position Error", perror, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log("Direction Error", derror, on_step=True, on_epoch=False, logger=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        val_loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss, z, py, ry = self._shared_step(batch, batch_idx)
        return test_loss

    def _shared_step(self, batch, batch_idx):
        x, py, ry = batch["images"], batch["pos_labels"], batch["rot_labels"]
        z = self.model(x)
        loss = self.loss(z[:, :, :3], py) + self.loss(z[:, :, 3:], ry)
        return loss, z, py, ry

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def _unscale(self, targets):
        device = targets.get_device()
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32).to(device)
        std = torch.tensor(self.scaler.scale_, dtype=torch.float32).to(device)
        targets = targets*std + mean
        mean.cpu(), std.cpu()
        return targets

    def _compute_errors(self, outputs):
        targets = self._unscale(torch.cat(outputs["targets"], dim=-1))
        preds = self._unscale(outputs["preds"])
        perror = torch.mean(torch.tensor([
            self.pos_error_train(preds[b, t, :3], targets[b, t, :3])
            for t in range(targets.shape[1]) for b in range(targets.shape[0])]))
        derror = torch.mean(torch.tensor([
            self.dir_error_train(preds[b, t, 3:], targets[b, t, 3:])
            for t in range(targets.shape[1]) for b in range(targets.shape[0])]))
        return perror, derror