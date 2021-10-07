import pathlib
import os
from pickle import load
import pandas as pd
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from .models import bronchonet, offsetnet
from .metrics import EuclideanDistance, NeedleError, DirectionError


def choose_model(model):
    switch = {
        "singletemporal": bronchonet.BronchoNetSingleTemporal(),
        "doubleearlytemporal": bronchonet.BronchoNetDoubleTemporalEarlyFusion(),
        "doublelatetemporal": bronchonet.BronchoNetDoubleTemporalLateFusion(),
        "doublelate": bronchonet.BronchoNetDoubleLateFusion(),
        "offsetnet": offsetnet.OffsetNet()
    }
    return switch.get(model, "Not an available model")


class BronchoModel(pl.LightningModule):

    def __init__(self, pred_folder="./data/cleaned/preds", lr=1e-4, model="singletemporal"):
        super().__init__()
        self.model = choose_model(model)
        self.loss = nn.MSELoss()
        parent_dir = pathlib.Path(__file__).parent.absolute()
        self.scaler = load(open(os.path.join(parent_dir, "data", "scaler.pkl"), "rb"))
        self.register_buffer("scaler_mean", torch.tensor(self.scaler.mean_, dtype=torch.float32))
        self.register_buffer("scaler_scale", torch.tensor(self.scaler.scale_, dtype=torch.float32))
        self.perror, self.derror, self.nerror = EuclideanDistance(), DirectionError(), NeedleError()
        self.pred_folder = pred_folder
        os.makedirs(pred_folder, exist_ok=True)
        self.lr = lr
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DRModel")
        parser.add_argument("--lr", dest="lr", type=float, default=1e-4)
        parser.add_argument("--pred-folder", dest="pred_folder", type=str, default="./data/cleaned/preds")
        parser.add_argument("--model", dest="model", type=str, default="singletemporal")
        return parent_parser

    def training_step(self, batch, batch_idx):
        loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        return {"loss": loss, "preds": z.detach(), "targets": torch.cat([py, ry], dim=-1)}

    def training_epoch_end(self, outputs):
        for output in outputs:
            self._compute_errors(output)
        self.log("Train Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        val_loss, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("val_loss", val_loss.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": val_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1)}

    def validation_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Val Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        test_loss, z, py, ry = self._shared_step(batch, batch_idx)
        return {"loss": test_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1)}

    def test_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset()
        for output in outputs:
            self._compute_errors(output)
            self._save_test_results(output)
        self.log("Test Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Test Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Test Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def _shared_step(self, batch, batch_idx):
        x, py, ry = batch["images"], batch["pos_labels"], batch["rot_labels"]
        z = self.model(x)
        loss = self.loss(z[:, :, :3], py) + 0.175*self.loss(z[:, :, 3:], ry)
        return loss, z, py, ry

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler_dict = {
            'scheduler': optim.lr_scheduler.ExponentialLR(
                optimizer,
                0.99
            ),
            'interval': 'epoch',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    def _unscale(self, targets):
        targets = targets*self.scaler_scale.type_as(targets) + self.scaler_mean.type_as(targets)
        return targets

    def _compute_errors(self, outputs):
        targets = self._unscale(outputs["targets"])
        preds = self._unscale(outputs["preds"])
        _ = [self.perror(preds[b, t, :3], targets[b, t, :3])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.derror(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.nerror(preds[b, t, :], targets[b, t, :])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]        

    def _save_test_results(self, outputs):
        targets = self._unscale(outputs["targets"])
        preds = self._unscale(outputs["preds"])
        for pred, target in zip(preds, targets):
            df = pd.DataFrame(columns=["shift_x", "shift_y", "shift_z", "qx", "qy", "qz",
                                       "gt_shift_x", "gt_shift_y", "gt_shift_z", "gt_qx", "gt_qy", "gt_qz"],
                              data=torch.cat((pred, target), dim=-1).cpu().numpy())
            df.to_csv(os.path.join(self.pred_folder, str(len(os.listdir(self.pred_folder))) + ".csv"))
