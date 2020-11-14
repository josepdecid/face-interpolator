from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pytorch_lightning import Trainer

from data import CelebADataModule
from models import ConvVAE
from utils import join_path


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--job_name', type=str)
    args = parser.parse_args()

    # TODO: Define config file
    dataset_root = join_path('datasets', 'CelebA')
    batch_size = 1
    num_workers = 0
    bottleneck_size = 100

    celebA_data_module = CelebADataModule(dataset_root, batch_size, num_workers)
    celebA_data_module.setup(stage='test')
    test_set = celebA_data_module.test_set
    model = ConvVAE.load_from_checkpoint('./output/train_albert_v1/train_albert_v1-epoch=29-val_loss=0.02.ckpt',
                                         bottleneck_size=bottleneck_size)
    model.eval()

    # Uncomment to do test_step
    # trainer = Trainer.from_argparse_args(args, checkpoint_callback=False, logger=False)
    # trainer.test(model, datamodule=celebA_data_module)

    image, attributes = test_set[0]
    image = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    decoded, mu, logvar = model(image)
    print("Encoded shape:", mu.shape)
    print("Decoded shape:", decoded.shape)

    # show images
    imshow(torchvision.utils.make_grid([image[0], decoded[0]]).detach())
