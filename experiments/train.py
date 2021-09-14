from pytorch_lightning.accelerators import accelerator
from BronchoTrack.models.bronchonet import BronchoNet
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from BronchoTrack.trainer import BronchoModel
from BronchoTrack.data.datasets import BronchoDataModule


def main(hparams):
    model = BronchoModel()
    trainer = Trainer(gpus=3, accelerator="ddp", log_every_n_steps=4)
    checkpoint_callback = checkpointing()
    trainer.callbacks = [checkpoint_callback]
    logger = TensorBoardLogger("logs", name="BronchoModel")
    trainer.logger = logger
    drData = BronchoDataModule(hparams.root, hparams.image_root, 4)
    drData.setup()
    if hparams.predict:
        model = model.load_from_checkpoint(hparams.ckpt)
        # model.predict(drData.test_dataloader())
    else:
        trainer.fit(model, drData)


def checkpointing():
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="bronchonet-{epoch:03d}-{val_loss:.5f}",
        save_top_k=1,
        mode="min",
    )
    return checkpoint_callback


def parse():
    parser = ArgumentParser()
    parser.add_argument("--root", dest="root", type=str)
    parser.add_argument("--image-root", dest="image_root", type=str)
    parser.add_argument("--predict", dest="predict", action="store_true")
    parser.add_argument("--ckpt", dest="ckpt", type=str)
    # TODO: batch_size, witdth, height, chcek how to build args in lightninh
    return parser


if __name__ == "__main__":

    # seed_everything(42, workers=True)
    parser = parse()
    args = parser.parse_args()
    main(args)
