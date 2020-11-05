from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from face_interpolator.data.celeba_dataset import CelebaDataset
from models.cnn_vae import ConvVAE

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # TODO: Define config file
    dataset_root = '../datasets/CelebA'
    batch_size = 64
    num_workers = 0

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = CelebaDataset(dataset_root, split='train', transform=transform)
    val_set = CelebaDataset(dataset_root, split='val', transform=transform)
    test_set = CelebaDataset(dataset_root, split='test', transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger = TensorBoardLogger('tb_logs')

    bottleneck_size = 40
    model = ConvVAE(bottleneck_size)
    trainer = Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, train_loader)
