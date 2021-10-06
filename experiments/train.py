from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from BronchoTrack.trainer import BronchoModel
from BronchoTrack.data.datasets import BronchoDataModule


def main(hparams):
    model = BronchoModel(pred_folder=hparams.pred_folder, lr=hparams.lr,
                         model=hparams.model)
    trainer = Trainer.from_argparse_args(hparams, accumulate_grad_batches=16, precision=16)
    checkpoint_callback = checkpointing(hparams.ckpt_name)
    trainer.callbacks = [checkpoint_callback,
                         LearningRateMonitor(logging_interval='epoch')]
    logger = TensorBoardLogger("logs", name="BronchoModel", version=args.ckpt_name)
    trainer.logger = logger
    drData = BronchoDataModule(hparams.root, hparams.image_root, hparams.batch_size)
    drData.setup()
    if hparams.predict:
        model = model.load_from_checkpoint(hparams.ckpt)
        trainer.test(model, drData)
    else:
        trainer.fit(model, drData)


def checkpointing(name):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=name + "-{epoch:03d}-{val_loss:.5f}",
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
    parser.add_argument("--ckpt-name", dest="ckpt_name", type=str, default="bronchonet")
    parser.add_argument("--batch-size", dest="batch_size", type=int,  default=16)
    return parser


if __name__ == "__main__":
    parser = parse()
    parser = BronchoModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
