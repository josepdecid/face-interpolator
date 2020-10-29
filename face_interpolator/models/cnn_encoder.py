import pytorch_lightning as pl
import torch.nn as nn


class CNNEncoder(pl.LightningModule):
    """
    Convolutional Encoder module.

    Args:
        - bottleneck_size (int): output size of the last linear layer.
        - channels (int, optional): number of channels of the decoded image.
    """

    def __init__(self, bottleneck_size, channels=3):
        super(CNNEncoder, self).__init__()

        self.bottle_neck_size = bottleneck_size

        self.encoder = nn.Sequential(
            # input is channels x 218 x 178
            nn.Conv2d(channels, 32, 7, stride=2, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # state size. 32 x 107 x 87
            nn.Conv2d(32, 64, 5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # state size. 64 x 53 x 43
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # state size. 128 x 26 x 21
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # state size. 256 x 13 x 11
            nn.Conv2d(256, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # state size. 512 x 6 x 5
            nn.Conv2d(512, 1024, (6, 5)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # nn.Sigmoid()
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, self.bottle_neck_size)
        self.fc22 = nn.Linear(512, self.bottle_neck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x.view(-1, 1024))
        return self.fc21(x), self.fc22(x)
