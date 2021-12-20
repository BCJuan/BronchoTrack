import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    BaseFinetuning, ModelCheckpoint, LearningRateMonitor,
    EarlyStopping, ModelPruning
    )

_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)
_MAX_EPOCHS = 75
_MIN_EPOCHS = 1

def create_amount(max):
    def amount(epoch):
        print("Pruning rate", max/(_MAX_EPOCHS - _MIN_EPOCHS), " and epoch ", epoch)
        if epoch < _MAX_EPOCHS and epoch > _MIN_EPOCHS:
            return max/(_MAX_EPOCHS - _MIN_EPOCHS)
        else:
            return 0.0
    return amount


def return_pruning_params(model, parameters=["weight"]):
    current_modules = [(n, m) for n, m in model.model.named_modules() if not isinstance(m, _MODULE_CONTAINERS)]
    parameters_to_prune = [
                (m, p) for p in parameters for n, m in current_modules if getattr(m, p, None) is not None and isinstance(m, nn.Conv2d)
                ]
    return parameters_to_prune


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


def pruning_callbacks(model, value):
    return [ModelPruning(
            pruning_fn="ln_structured",
            parameters_to_prune=return_pruning_params(model),
            amount=create_amount(value),
            use_global_unstructured=False,
            pruning_norm=1,
            pruning_dim=1,
            parameter_names=['weight'],
            use_lottery_ticket_hypothesis=False,
            prune_on_train_epoch_end=True
        )]


def build_callbacks(version_name, model, prune=None):
    checkpoint_callback = checkpointing(version_name)
    callbacks =  [checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor="val_loss", patience=25, min_delta=0.005)]
    if prune:
        callbacks = callbacks + pruning_callbacks(model, prune)
        print(callbacks[-1]._parameters_to_prune)
    return callbacks
