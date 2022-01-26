import pathlib
import os
from pickle import load
import pandas as pd
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from tqdm import tqdm
from .models import bronchonet, offsetnet, deependovo
from .metrics import (
    EuclideanDistance, NeedleError, DirectionError,
    CosMetric, QuatMetric, MSE
)
from .losses import (
    CosLoss, DirectionLoss, QuaternionDistanceLoss, EuclideanDistanceLoss,
    UpperBoundTeacherLoss
)


def choose_model(model):
    switch = {
        "singletemporal": bronchonet.BronchoNetSingleTemporal(),
        "doubleearlytemporal": bronchonet.BronchoNetDoubleTemporalEarlyFusion(),
        "doublelatetemporal": bronchonet.BronchoNetDoubleTemporalLateFusion(),
        "doublelate": bronchonet.BronchoNetDoubleLateFusion(),
        "offsetnet": offsetnet.OffsetNet(),
        "doublelate3d": bronchonet.BronchoNetDoubleLate3DFusion(),
        "doublelateconvtemporal": bronchonet.BronchoNetDoubleTemporalConvLateFusion(),
        "deependovo": deependovo.DeepEndovo()
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


class BronchoModel(pl.LightningModule):

    def __init__(self, pred_folder="./data/cleaned/preds", lr=1e-4, model="singletemporal", rot_loss="mse", pos_loss="mse",
                 distill_teacher=None, teacher_alpha=0.2, distill_student=None, student_alpha=0.1):
        super().__init__()
        self.model = choose_model(model)
        self.distill_teacher = distill_teacher
        self.distill_student = distill_student
        if distill_teacher:
            self.teacher = BronchoModel.load_from_checkpoint(distill_teacher)
            self.teacher.eval()
            self.teacher.freeze()
            self.teacherloss = UpperBoundTeacherLoss(choose_pos_loss(pos_loss), teacher_alpha)
            self.teacherloss_1 = UpperBoundTeacherLoss(choose_rot_loss(rot_loss), teacher_alpha)
            if distill_student:
                self.student = BronchoModel.load_from_checkpoint(distill_student)
                self.student.eval()
                self.student.freeze()
                self.studentloss = UpperBoundTeacherLoss(choose_pos_loss(pos_loss), student_alpha)
                self.studentloss_1 = UpperBoundTeacherLoss(choose_rot_loss(rot_loss), student_alpha)
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

    def training_step(self, batch, batch_idx):
        loss, loss_p, loss_r, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        self.log("train_loss_position", loss_p.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        self.log("train_loss_rotation", loss_r.detach(), on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=True)
        return {"loss": loss, "preds": z.detach(), "targets": torch.cat([py, ry], dim=-1)}

    def training_epoch_end(self, outputs):
        self.perror.reset(), self.derror.reset(), self.nerror.reset(), self.cos.reset()
        self.cosx.reset(), self.cosy.reset(), self.cosz.reset, self.mse.reset(), self.quat.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Train Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train MSE Angle Error", self.mse, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS x Angle Error", self.cosx, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS y Angle Error", self.cosy, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS z Angle Error", self.cosz, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train COS Angle Error", self.cos, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Train Quat Angle Error", self.quat, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        val_loss, val_loss_p, val_loss_r, z, py, ry = self._shared_step(batch, batch_idx)
        self.log("val_loss", val_loss.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_position", val_loss_p.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss_rotation", val_loss_r.detach(), on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": val_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1)}

    def validation_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset(), self.cos.reset()
        self.cosx.reset(), self.cosy.reset(), self.cosz.reset, self.mse.reset(), self.quat.reset()
        for output in outputs:
            self._compute_errors(output)
        self.log("Val Position Error", self.perror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Direction Error", self.derror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Needle Error", self.nerror, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val MSE Angle Error", self.mse, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS x Angle Error", self.cosx, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS y Angle Error", self.cosy, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS z Angle Error", self.cosz, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val COS Angle Error", self.cos, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)
        self.log("Val Quat Angle Error", self.quat, on_step=False, on_epoch=True, logger=True, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        test_loss, loss_p, loss_r, z, py, ry = self._shared_step(batch, batch_idx)
        return {"loss": test_loss, "preds": z, "targets": torch.cat([py, ry], dim=-1), "filenames": batch["filename"]}

    def test_epoch_end(self, outputs) -> None:
        self.perror.reset(), self.derror.reset(), self.nerror.reset()
        for output in tqdm(outputs, total=len(outputs)):
            self._compute_errors(output)
            self._save_test_results(output)

    def _shared_step(self, batch, batch_idx):
        x, py, ry = batch["images"], batch["pos_labels"], batch["rot_labels"]
        z = self.model(x)
        loss_p = self.loss(z[:, :, :3], py)
        loss_r = self.loss1(z[:, :, 3:], ry)
        if self.distill_teacher:
            z_t = self.teacher.model(x)
            loss_p = self.teacherloss(z_t[:, :, :3], py, z[:, :, :3], loss_p)
            loss_r = self.teacherloss_1(z_t[:, :, 3:], ry, z[:, :, 3:], loss_r)
            if self.distill_student:
                z_s = self.student.model(x)
                loss_p = self.studentloss(z_s[:, :, :3], py, z[:, :, :3], loss_p)
                loss_r = self.studentloss_1(z_s[:, :, 3:], ry, z[:, :, 3:], loss_r)                
        loss = loss_p + loss_r
        return loss, loss_p, loss_r, z, py, ry

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler_dict = {
            'scheduler': optim.lr_scheduler.ExponentialLR(
                optimizer,
                0.9995
            ),
            'interval': 'epoch',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

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
        for pred, target, filename in zip(preds, targets, outputs["filenames"]):
            df = pd.DataFrame(columns=["shift_x", "shift_y", "shift_z", "Rx_dif", "Ry_dif", "Rz_dif",
                                       "gt_shift_x", "gt_shift_y", "gt_shift_z", "gt_Rx_dif", "gt_Ry_dif", "gt_Rz_dif"],
                              data=torch.cat((pred, target), dim=-1).cpu().numpy())
            df.to_csv(os.path.join(self.pred_folder, os.path.basename(filename)))
