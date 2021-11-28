import random
import numpy as np
import torch
from pytorch_lightning.callbacks import (
    BaseFinetuning, ModelCheckpoint, LearningRateMonitor,
    EarlyStopping
    )


def checkpointing(name):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/" + name,
        filename=name,
        save_top_k=1,
        mode="min",
    )
    return checkpoint_callback


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BackboneFineTuning(BaseFinetuning):

    def __init__(self, unfreeze_at_epoch=10, single=False):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.single = single

    def freeze_before_training(self, pl_module):
        if self.single:
            self.freeze(pl_module.model.backbone)
        else:
            self.freeze([pl_module.model.backbone_t, pl_module.model.backbone_t1])

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch == self._unfreeze_at_epoch:
            if self.single:
                modules = pl_module.model.backbone
            else:
                modules = [pl_module.model.backbone_t, pl_module.model.backbone_t1]
            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=False,
            )


def build_callbacks(version_name, model):
    checkpoint_callback = checkpointing(version_name)
    return [checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor="val_loss", patience=20)]
            # BackboneFineTuning(20, True if "single" in model else False)]
