from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from face_interpolator.data.celeba_dataset import CelebADataModule
from face_interpolator.utils.system import join_path
from face_interpolator.models.model_factory import ModelFactory


def train():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--job-name', type=str)
    parser.add_argument('--bottleneck', type=int, default=128)
    parser.add_argument('--model', type=str, choices=ModelFactory.available_models())
    parser.add_argument('--attribute-size', type=int, default=128)
    args = parser.parse_args()

    model = ModelFactory(args)

    # TODO: Define config file
    dataset_root = join_path('datasets', 'CelebA')
    batch_size = 64
    num_workers = 0

    celebA_data_module = CelebADataModule(dataset_root, batch_size, num_workers)

    logger = TensorBoardLogger(join_path('output', args.job_name, 'tb_logs'), name='')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=join_path('output', args.job_name, 'checkpoints'),
        filename=args.job_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=celebA_data_module)
