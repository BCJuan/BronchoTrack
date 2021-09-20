import pathlib
import os
from pickle import load
import pandas as pd
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from .models import bronchonet
from .metrics import EuclideanDistance, NeedleError, DirectionError


class BronchoModel(pl.LightningModule):

    def __init__(self, pred_folder="./data/cleaned/preds"):
        super().__init__()
        self.model = bronchonet.BronchoNetSingleTemporal()
        self.loss = nn.MSELoss()
        parent_dir = pathlib.Path(__file__).parent.absolute()
        self.scaler = load(open(os.path.join(parent_dir, "data", "scaler.pkl"), "rb"))
        self.perror, self.derror, self.nerror = EuclideanDistance(), DirectionError(), NeedleError()
        self.pred_folder = pred_folder
        os.makedirs(pred_folder, exist_ok=True)

    def training_step(self, batch, batch_idx):
        loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        return {"loss": loss, "preds": z.detach(), "targets": (py, ry)}

    def training_step_end(self, outputs):
        self._compute_errors(outputs)
        self.log("Train Position Error", self.perror, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log("Train Direction Error", self.derror, on_step=True, on_epoch=False, logger=True, prog_bar=False)
        self.log("Train Needle Error", self.nerror, on_step=True, on_epoch=False, logger=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        val_loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return {"loss": val_loss, "preds": z, "targets": (py, ry)}

    def validation_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Val Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log("Val Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log("Val Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        test_loss, z, py, ry = self._shared_step(batch, batch_idx)
        return {"loss": test_loss, "preds": z, "targets": (py, ry)}

    def test_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset()
        for output in outputs:
            self._compute_errors(output)
            self._save_test_results(output)
        self.log("Test Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log("Test Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log("Test Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False)

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
        _ = [self.perror(preds[b, t, :3], targets[b, t, :3])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.derror(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.nerror(preds[b, t, :], targets[b, t, :])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]        

    def _save_test_results(self, outputs):
        targets = self._unscale(torch.cat(outputs["targets"], dim=-1))
        preds = self._unscale(outputs["preds"])
        for pred, target in zip(preds, targets):
            df = pd.DataFrame(columns=["shift_x", "shift_y", "shift_z", "qx", "qy", "qz",
                                       "gt_shift_x", "gt_shift_y", "gt_shift_z", "gt_qx", "gt_qy", "gt_qz"],
                              data=torch.cat((pred, target), dim=-1).cpu().numpy())
            df.to_csv(os.path.join(self.pred_folder, str(len(os.listdir(self.pred_folder))) + ".csv"))
