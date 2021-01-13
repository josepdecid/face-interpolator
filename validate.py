from argparse import ArgumentParser

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pytorch_lightning import Trainer

from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.data import CelebADataModule
from face_interpolator.utils import join_path
from face_interpolator.utils.unormalize import UnNormalize
from models.vanilla_vae import ConvVAE


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title("VAE")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--job_name', type=str)
    args = parser.parse_args()

    dataset_root = join_path('datasets', 'CelebA')
    batch_size = 1
    num_workers = 0
    bottleneck_size = 256

    celebA_data_module = CelebADataModule(dataset_root, batch_size, num_workers)
    celebA_data_module.setup(stage='test')
    test_set = celebA_data_module.test_set
    model = ConvVAE.load_from_checkpoint('./output/run03-epoch=103-val_loss=2239237.75.ckpt',
                                         bottleneck_size=bottleneck_size)
    model.eval()
    torch.set_grad_enabled(False)

    # Uncomment to do test_step
    # trainer = Trainer.from_argparse_args(args, checkpoint_callback=False, logger=False)
    # trainer.test(model, datamodule=celebA_data_module)

    rand_sample = random.randint(0, len(test_set)-1)
    image, attributes = test_set[rand_sample]
    image = torch.reshape(image, (1, image.shape[-3], image.shape[-2], image.shape[-1]))
    decoded, mu, logvar = model(image)
    print("Encoded shape:", mu.shape)
    print("Decoded shape:", decoded.shape)

    unorm = UnNormalize(mean=MEAN, std=STD)
    unorm(image[0])
    unorm(decoded[0])

    # show images
    imshow(torchvision.utils.make_grid([image[0], decoded[0]]).detach())