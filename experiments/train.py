import os
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from BronchoTrack.trainer import BronchoModel
from BronchoTrack.data.datasets import BronchoDataModule
from pytorch_lightning.plugins import DDPPlugin


def main(hparams):
    model = BronchoModel(pred_folder=hparams.pred_folder, lr=hparams.lr,
                         model=hparams.model, loss=hparams.loss)
    version_name = "_".join([hparams.ckpt_name, hparams.model])
    # check_val_every = 5
    trainer = Trainer.from_argparse_args(hparams,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    accumulate_grad_batches=16,
                    deterministic=True,
                    # max_epochs=185,
                    log_every_n_steps=10,
                    # flush_logs_every_n_steps=150,
                    # check_val_every_n_epoch=check_val_every
                    )
    checkpoint_callback = checkpointing(version_name)
    trainer.callbacks = [checkpoint_callback,
                         LearningRateMonitor(logging_interval='epoch'),
                         EarlyStopping(monitor="val_loss", patience=50)]
    logger = None if hparams.predict else TensorBoardLogger("logs", name="BronchoModel", version=version_name)
    trainer.logger = logger
    drData = BronchoDataModule(hparams.root, hparams.image_root, hparams.batch_size, augment=hparams.augment, only_val=hparams.only_val)
    drData.setup()
    if hparams.predict:
        if hparams.ckpt:
            model = model.load_from_checkpoint(hparams.ckpt)
        else:
            model = model.load_from_checkpoint(os.path.join("./checkpoints", version_name, version_name + ".ckpt"))
        model.pred_folder = hparams.pred_folder
        trainer.test(model, drData)
    else:
        trainer.fit(model, drData)


def checkpointing(name):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints/" + name,
        filename=name,
        save_top_k=1,
        mode="min",
    )
    return checkpoint_callback


def parse():
    parser = ArgumentParser()
    parser.add_argument("--root", dest="root", type=str)
    parser.add_argument("--image-root", dest="image_root", type=str)
    parser.add_argument("--predict", dest="predict", action="store_true", default=False)
    parser.add_argument("--ckpt", dest="ckpt", type=str)
    parser.add_argument("--ckpt-name", dest="ckpt_name", type=str, default="bronchonet")
    parser.add_argument("--batch-size", dest="batch_size", type=int,  default=16)
    parser.add_argument("--augment", dest="augment", action="store_true", default=False)
    parser.add_argument("--only-val", dest="only_val", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    seed_everything(42, workers=True)
    parser = parse()
    parser = BronchoModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
