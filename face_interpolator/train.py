from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import os
from face_interpolator.data.celeba_dataset import CelebADataModule
from .models.cnn_vae import ConvVAE


def train():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # TODO: Define config file
    dataset_root = os.path.join('..', 'datasets', 'CelebA')
    batch_size = 64
    num_workers = 0

    celebA_data_module = CelebADataModule(dataset_root, batch_size, num_workers)

    logger = TensorBoardLogger('tb_logs')

    bottleneck_size = 40
    model = ConvVAE(bottleneck_size)
    trainer = Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, datamodule=celebA_data_module)


if __name__ == '__main__':
    train()
