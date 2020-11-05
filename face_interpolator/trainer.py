from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from face_interpolator.data.celeba_dataset import CelebADataModule
from .models.cnn_vae import ConvVAE
from .utils.system import join_path


def train():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--job_name', type=str)
    args = parser.parse_args()

    # TODO: Define config file
    dataset_root = os.path.join('..', 'datasets', 'CelebA')
    batch_size = 64
    num_workers = 0
    bottleneck_size = 40

    celebA_data_module = CelebADataModule(dataset_root, batch_size, num_workers)

    logger = TensorBoardLogger(join_path('..', 'output', args.job_name, 'tb_logs'), name='')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=join_path('..', 'output', args.job_name, 'checkpoints'),
        filename=args.job_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')

    model = ConvVAE(bottleneck_size)
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=celebA_data_module)
