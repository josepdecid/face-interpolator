from face_interpolator.models.model import AutoEncoderModel
import torch
from torch import nn
from typing import Tuple, Union, Optional


class _ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size, stride: int = 2, n_layers: int = 3):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = 1

        def build_layer(channels_in, channels_out, kernel_size, stride, padding):
            return nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                                           padding=padding, stride=stride),
                                 nn.BatchNorm2d(channels_out),
                                 nn.LeakyReLU())
        layers = [build_layer(self.channels_in, self.channels_out, self.kernel_size, self.stride, self.padding)]
        for i in range(n_layers-1):
            layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, 1, self.padding))

        self.layers = nn.ModuleList(layers)

        self.shortcut = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        x = x + shortcut
        return x


class _UpsamplingResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: Union[int, Tuple[int, int]],
                 padding: Optional[int] = None, n_layers: int = 3, output_size: Optional[Tuple[int, int]] = None):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_layers = n_layers
        self.output_size = output_size
        self.padding = 1
        self.stride = 2

        def build_layer(channels_in, channels_out, kernel_size, stride, padding):
            return nn.Sequential(nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size,
                                                    padding=padding, stride=stride),
                                 nn.BatchNorm2d(channels_out),
                                 nn.LeakyReLU())
        layers = [build_layer(self.channels_in, self.channels_out, self.kernel_size, 2, padding)]
        for i in range(n_layers-1):
            layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, 1, padding))

        self.layers = nn.ModuleList(layers)

        self.shortcut = nn.ConvTranspose2d(self.channels_in, self.channels_out, kernel_size=1,
                                           stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        x = x + shortcut
        return x


class _ResidualEncoder(nn.Module):
    def __init__(self, bottleneck_size: int, channels: int, dropout: float):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channels = channels
        self.dropout = dropout

        blocks = [_ResidualBlock(channels_in=self.channels, channels_out=32, kernel_size=7,
                                 stride=2),
                  _ResidualBlock(channels_in=64, channels_out=128, kernel_size=5,
                                 stride=2),
                  _ResidualBlock(channels_in=128, channels_out=256, kernel_size=3,
                                 stride=2),
                  _ResidualBlock(channels_in=256, channels_out=512, kernel_size=3,
                                 stride=2),
                  _ResidualBlock(channels_in=512, channels_out=1024, kernel_size=(6, 5),
                                 stride=2),
                  ]

        self.blocks = nn.ModuleList(blocks)

        def build_linear(in_, out):
            return nn.Sequential(nn.Linear(in_, out), nn.BatchNorm1d(out), nn.LeakyReLU(), nn.Dropout(self.dropout))

        self.fc = build_linear(1024, 512)
        self.fc_mu = build_linear(512, self.bottleneck_size)
        self.fc_log_var = build_linear(512, self.bottleneck_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class _ResidualDecoder(nn.Module):
    def __init__(self, bottleneck_size: int, target_channels: int, dropout: float):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.target_channels = target_channels
        self.dropout = dropout

        def build_linear(in_, out):
            return nn.Sequential(nn.Linear(in_, out), nn.BatchNorm1d(out), nn.LeakyReLU(), nn.Dropout(self.dropout))

        self.fc1 = build_linear(self.bottleneck_size, 512)
        self.fc2 = build_linear(512, 1024)

        blocks = [_UpsamplingResidualBlock(channels_in=1024, channels_out=512, kernel_size=(6, 5), output_size=(6, 5)),
                  _UpsamplingResidualBlock(channels_in=512, channels_out=256, kernel_size=3, output_size=(13, 11)),
                  _UpsamplingResidualBlock(channels_in=256, channels_out=128, kernel_size=3, output_size=(26, 21)),
                  _UpsamplingResidualBlock(channels_in=128, channels_out=64, kernel_size=5, padding=1,
                                           output_size=(53, 43)),
                  _UpsamplingResidualBlock(channels_in=64, channels_out=32, kernel_size=5, output_size=(107, 87)),
                  _UpsamplingResidualBlock(channels_in=32, channels_out=self.target_channels, kernel_size=7,padding=1,
                                           output_size=(218, 178))
                  ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1024, 1, 1)
        for block in self.blocks:
            x = block(x)
        return x


class ResidualVAE(nn.Module):
    def __init__(self, bottleneck_size: int = 128, channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channels = channels
        self.dropout = dropout
        self.encoder = _ResidualEncoder(self.bottleneck_size, self.channels, self.dropout)
        self.decoder = _ResidualDecoder(self.bottleneck_size, target_channels=self.channels, dropout=self.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self._reparameterize(mu, log_var)
        decoded = self.decode(z)
        return decoded, mu, log_var

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        return mu, log_var

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


if __name__ == '__main__':
    '''
    x = torch.zeros(16, 256, 64, 32)
    block = _ResidualBlock(channels_in=256, channels_out=128, decrease_size_factor=2)
    print(block(x).shape)
    x = torch.zeros(16, 3, 64, 32)
    encoder = _ResidualEncoder(bottleneck_size=128, channels=3, initial_channels=32, n_blocks=4, factor=2,
                               dropout=0.5, height=64, width=32)
    encoder_out = encoder(x)
    print(encoder_out[0].shape)
    '''

    x = torch.zeros(16, 3, 218, 178)
    #encoder = _ResidualEncoder(bottleneck_size=128, channels=3, initial_channels=32, n_blocks=4, factor=2,
    #                           dropout=0.5, height=218, width=178)
    #encoder(x)
    model = ResidualVAE()
    assert model(x)[0].shape == x.shape
