import pathlib
import os
from pickle import load
from numpy.core.fromnumeric import choose
import pandas as pd
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from tqdm import tqdm
from .models import bronchonet, offsetnet
from .metrics import (
    EuclideanDistance, NeedleError, DirectionError,
    CosMetric, QuatMetric, MSE
)
from .losses import CosLoss, DirectionLoss, QuaternionDistanceLoss, EuclideanDistanceLoss


def choose_model(model):
    switch = {
        "singletemporal": bronchonet.BronchoNetSingleTemporal(),
        "doubleearlytemporal": bronchonet.BronchoNetDoubleTemporalEarlyFusion(),
        "doublelatetemporal": bronchonet.BronchoNetDoubleTemporalLateFusion(),
        "doublelate": bronchonet.BronchoNetDoubleLateFusion(),
        "offsetnet": offsetnet.OffsetNet(),
        "doublelate3d": bronchonet.BronchoNetDoubleLate3DFusion()
    }
    return switch.get(model, "Not an available model")


def choose_rot_loss(loss):
    switch = {
        "mse": nn.MSELoss(),
        "cos": CosLoss(),
        "direction": DirectionLoss(),
        "quaternion": QuaternionDistanceLoss()
    }
    return switch.get(loss, "Not an available loss")


def choose_pos_loss(loss):
    switch = {
        "mse": nn.MSELoss(),
        "euclidean": EuclideanDistanceLoss(),
    }
    return switch.get(loss, "Not an available loss")    


class BronchoModelDistilled(pl.LightningModule):

    def __init__(self, pred_folder="./data/cleaned/preds", lr=1e-4, model="singletemporal", rot_loss="mse", pos_loss="mse"):
        super().__init__()
        self.model = choose_model(model)
        self.model1 = choose_model(model)
        self.model2 = choose_model(model)
        self.model3 = choose_model(model)
        self.model4 = choose_model(model)
        self.loss = choose_pos_loss(pos_loss)
        self.loss1 = choose_rot_loss(rot_loss)
        self.perror, self.derror, self.nerror = EuclideanDistance(), DirectionError(), NeedleError()
        self.mse, self.cosx, self.cosy, self.cosz = MSE(), CosMetric(0), CosMetric(1), CosMetric(2)
        self.cos, self.quat = CosMetric(), QuatMetric()
        self.lr = lr
        self.pred_folder = pred_folder
        os.makedirs(pred_folder, exist_ok=True)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DRModel")
        parser.add_argument("--lr", dest="lr", type=float, default=1e-4)
        parser.add_argument("--pred-folder", dest="pred_folder", type=str, default="./data/cleaned/preds")
        parser.add_argument("--model", dest="model", type=str, default="singletemporal")
        parser.add_argument("--rot-loss", dest="rot_loss", type=str, default="mse")
        parser.add_argument("--pos-loss", dest="pos_loss", type=str, default="mse")
        return parent_parser

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, loss_p, loss_r, z, py, ry = self._shared_step(batch, batch_idx, optimizer_idx)
        self.log("train_loss_" + str(optimizer_idx), loss.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        self.log("train_loss_position_" + str(optimizer_idx), loss_p.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        self.log("train_loss_rotation_" + str(optimizer_idx), loss_r.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        return {"loss": loss, "preds": z.detach(), "targets": torch.cat([py, ry], dim=-1), "optimizer_idx": str(optimizer_idx)}

    def training_epoch_end(self, outputs):
        self.perror.reset(), self.derror.reset(), self.nerror.reset(), self.cos.reset()
        self.cosx.reset(), self.cosy.reset(), self.cosz.reset, self.mse.reset(), self.quat.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Train Position Error " + output["optimizer_idx"], self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Direction Error " + output["optimizer_idx"], self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Needle Error " + output["optimizer_idx"], self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train MSE Angle Error " + output["optimizer_idx"], self.mse, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS x Angle Error " + output["optimizer_idx"], self.cosx, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS y Angle Error " + output["optimizer_idx"], self.cosy, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS z Angle Error " + output["optimizer_idx"], self.cosz, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS Angle Error " + output["optimizer_idx"], self.cos, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Quat Angle Error " + output["optimizer_idx"], self.quat, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx, optimizer_idx):
        val_loss, val_loss_p, val_loss_r, z, py, ry = self._shared_step(batch, batch_idx, optimizer_idx)
        self.log("val_loss_" + str(optimizer_idx), val_loss.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_position_" + str(optimizer_idx), val_loss_p.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_rotation_" + str(optimizer_idx), val_loss_r.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": val_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1), "optimizer_idx": str(optimizer_idx)}

    def validation_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset(), self.cos.reset()
        self.cosx.reset(), self.cosy.reset(), self.cosz.reset, self.mse.reset(), self.quat.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Val Position Error " + output["optimizer_idx"], self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Direction Error " + output["optimizer_idx"], self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Needle Error " + output["optimizer_idx"], self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val MSE Angle Error " + output["optimizer_idx"], self.mse, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS x Angle Error " + output["optimizer_idx"], self.cosx, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS y Angle Error " + output["optimizer_idx"], self.cosy, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS z Angle Error " + output["optimizer_idx"], self.cosz, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS Angle Error " + output["optimizer_idx"], self.cos, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Quat Angle Error " + output["optimizer_idx"], self.quat, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx, optimizer_idx):
        test_loss, loss_p, loss_r, z, py, ry = self._shared_step(batch, batch_idx, optimizer_idx)
        return {"loss": test_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1), "filenames": batch["filename"], "optimizer_idx": str(optimizer_idx)}

    def test_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset(), self.cos.reset()
        self.cosx.reset(), self.cosy.reset(), self.cosz.reset, self.mse.reset(), self.quat.reset()
        for output in tqdm(outputs, total=len(outputs)):
            self._compute_errors(output)
            self._save_test_results(output)

    def _shared_step(self, batch, batch_idx, optimizer_idx):
        x, py, ry = batch["images"], batch["pos_labels"], batch["rot_labels"]
        if optimizer_idx == 0:
            z = self.model(x)
        elif optimizer_idx == 1:
            z = self.model1(x)
        elif optimizer_idx == 2:
            z = self.model2(x)
        elif optimizer_idx == 3:
            z = self.model3(x)
        else:
            z = self.model4(x)      
        loss_p = self.loss(z[:, :, :3], py)
        loss_r = self.loss1(z[:, :, 3:], ry)
        loss = loss_p + loss_r
        return loss, loss_p, loss_r, z, py, ry

    def configure_optimizers(self):
        optimizers = [
            optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=self.lr)
            for modelo in [self.model, self.model1, self.model2, self.model3, self.model4]]
        scheduler_dicts = {{
            'scheduler': optim.lr_scheduler.ExponentialLR(
                optimizer,
                0.9995
            ),
            'interval': 'epoch',
        } for optimizer in optimizers}
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
            for optimizer, scheduler_dict in zip(optimizers, scheduler_dicts)
        )

    def _compute_errors(self, outputs):
        targets = outputs["targets"]
        preds = outputs["preds"]
        _ = [self.perror(preds[b, t, :3], targets[b, t, :3])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.derror(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.nerror(preds[b, t, :], targets[b, t, :])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.mse(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.cosx(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.cosy(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.cosz(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.cos(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]
        _ = [self.quat(preds[b, t, 3:], targets[b, t, 3:])
             for t in range(targets.shape[1]) for b in range(targets.shape[0])]

    def _save_test_results(self, outputs):
        targets = outputs["targets"]
        preds = outputs["preds"]
        for pred, target, filename, opt_idx in zip(preds, targets, outputs["filenames"], outputs["optimizer_idx"]):
            df = pd.DataFrame(columns=["shift_x", "shift_y", "shift_z", "Rx_dif", "Ry_dif", "Rz_dif",
                                       "gt_shift_x", "gt_shift_y", "gt_shift_z", "gt_Rx_dif", "gt_Ry_dif", "gt_Rz_dif"],
                              data=torch.cat((pred, target), dim=-1).cpu().numpy())
            df.to_csv(
                os.path.join(
                    self.pred_folder,
                    os.path.splitext(os.path.basename(filename))[0] + "_model_" + opt_idx + ".csv"
                    )
                )

