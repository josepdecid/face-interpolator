import pytorch_lightning as pl
import torch.nn as nn


class MyConvTranspose2d(pl.LightningModule):
    """
    Custom Convolutional Transpose 2D layer which allows specifying the output size.

    Args:
        - conv (nn.ConvTranspose2d): Instance of a ConvTranspose2d layer.
        - output_size (Tuple, optional): Tuple indicating the desired output size of the layer.

    Attributes:
        - conv (nn.ConvTranspose2d): Where the conv arg is stored.
        - output_size (Tuple): Where the output_size arg is stored.
    """

    def __init__(self, conv: nn.ConvTranspose2d, output_size=None):
        super(MyConvTranspose2d, self).__init__()
        self.conv = conv
        self.output_size = output_size

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class CNNDecoder(pl.LightningModule):
    """
    Convolutional Decoder module.

    Args:
        - bottleneck_size (int): input size of the first linear layer.
        - channels (int, optional): number of channels of the decoded image.
    """

    def __init__(self, bottleneck_size, channels=3):
        super(CNNDecoder, self).__init__()

        self.bottle_neck_size = bottleneck_size

        self.fc3 = nn.Linear(self.bottle_neck_size, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.decoder = nn.Sequential(
            MyConvTranspose2d(nn.ConvTranspose2d(1024, 512, (6, 5)), output_size=(6, 5)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # state size. 512 x 6 x 5
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(512, 256, 3, stride=2), output_size=(13, 11)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # state size. 256 x 13 x 11
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1), output_size=(26, 21)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # state size. 128 x 26 x 21
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(128, 64, 5, stride=2, padding=1), output_size=(53, 43)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # state size. 64 x 53 x 43
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1), output_size=(107, 87)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # state size. 32 x 107 x 87
            # nn.MaxUnpool2d(2, 2),
            MyConvTranspose2d(nn.ConvTranspose2d(32, channels, 7, stride=2, padding=1), output_size=(218, 178)),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 1024, 1, 1)
        x = self.decoder(x)
        return x
