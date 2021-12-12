import random
import numpy as np
import torch
from pytorch_lightning.callbacks import (
    BaseFinetuning, ModelCheckpoint, LearningRateMonitor,
    EarlyStopping, ModelPruning
    )


params_prune = [(model.model1, "weight"), (model.model2, "weight"), (model.model3, "weight"), (model.model4, "weight")]


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


def pruning_callbacks():
    return [
        ModelPruning(
            pruning_fn="ln_structured`",
            parameters_to_prune=params,
            amount=qq,
            use_global_unstructured=False,
            pruning_norm=1
        ) for params, qq in zip(params_prune, np.arange(0.2, 1.0, 0.2))]


def build_callbacks(version_name, model, distill=None):
    checkpoint_callback = checkpointing(version_name)
    callbacks =  [checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor="val_loss", patience=25, min_delta=0.005)]
            # BackboneFineTuning(20, True if "single" in model else False)]
    if distill:
        callbacks = callbacks + pruning_callbacks()
    return callbacks
