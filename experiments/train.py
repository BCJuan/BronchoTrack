import os
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from BronchoTrack.trainer import BronchoModel
from BronchoTrack.data.datasets import BronchoDataModule
from BronchoTrack.utils import build_callbacks
import torch


def main(hparams):
    model = BronchoModel(pred_folder=hparams.pred_folder, lr=hparams.lr,
                         model=hparams.model, rot_loss=hparams.rot_loss,
                         pos_loss=hparams.pos_loss, distill_teacher=hparams.distill_teacher,
                         teacher_alpha=hparams.teacher_alpha, distill_student=hparams.distill_student,
                         student_alpha=hparams.student_alpha)
    version_name = "_".join([hparams.ckpt_name, hparams.model])

    if hparams.restore:
        restore_check = os.path.join("./checkpoints", version_name, version_name + ".ckpt")
    else:
        restore_check = None
    trainer = Trainer.from_argparse_args(hparams,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    accumulate_grad_batches=64,
                    deterministic=True,
                    log_every_n_steps=1,
                    resume_from_checkpoint=restore_check,
                    min_epochs=75
                    )
    
    trainer.callbacks = build_callbacks(version_name, model, hparams.prune)
    logger = None if hparams.predict else TensorBoardLogger("logs", name="BronchoModel", version=version_name)
    trainer.logger = logger
    drData = BronchoDataModule(hparams.root, hparams.image_root, hparams.batch_size, augment=hparams.augment)
    drData.setup()
    if hparams.predict:
        if hparams.ckpt:
            model = model.load_from_checkpoint(hparams.ckpt)
        else:
            model = model.load_from_checkpoint(checkpoint_path=os.path.join("checkpoints", version_name, version_name + ".ckpt"))
        model.pred_folder = hparams.pred_folder
        print(sum([p.nonzero().size(0) for p in model.model.parameters()]))
        trainer.test(model, drData)
    else:
        trainer.fit(model, drData)


def parse():
    parser = ArgumentParser()
    parser.add_argument("--root", dest="root", type=str)
    parser.add_argument("--image-root", dest="image_root", type=str)
    parser.add_argument("--predict", dest="predict", action="store_true", default=False)
    parser.add_argument("--ckpt", dest="ckpt", type=str)
    parser.add_argument("--ckpt-name", dest="ckpt_name", type=str, default="bronchonet")
    parser.add_argument("--batch-size", dest="batch_size", type=int,  default=16)
    parser.add_argument("--augment", dest="augment", action="store_true", default=False)
    parser.add_argument("--restore", dest="restore", action="store_true", default=False)
    parser.add_argument("--prune", dest="prune", type=float, default=None)
    parser.add_argument("--distill-teacher", dest="distill_teacher", type=str, default=None, help="checkpoint path for the teacher")
    parser.add_argument("--teacher-alpha", dest="teacher_alpha", type=float, default=0.2)
    parser.add_argument("--distill-student", dest="distill_student", type=str, default=None, help="checkpoint path for the helping student")
    parser.add_argument("--student-alpha", dest="student_alpha", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    seed_everything(42, workers=True)
    parser = parse()
    parser = BronchoModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
