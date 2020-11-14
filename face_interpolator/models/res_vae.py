from face_interpolator.models.model import AutoEncoderModel
import torch
from torch import nn
from typing import Tuple


class _ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, n_layers: int = 3, kernel_size: int = 5,
                 decrease_size_factor: int = 2):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.decrease_size_factor = decrease_size_factor

        def build_layer(channels_in, channels_out, kernel_size, decrease_size_factor):
            return nn.Sequential(nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                                           padding=kernel_size//2, stride=decrease_size_factor),
                                 nn.BatchNorm2d(channels_out),
                                 nn.LeakyReLU())
        layers = [build_layer(self.channels_in, self.channels_out, self.kernel_size, self.decrease_size_factor)]
        for i in range(n_layers-1):
            layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, 1))
         # layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, self.decrease_size_factor))

        self.layers = nn.ModuleList(layers)

        self.shortcut = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=self.decrease_size_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        x = x + shortcut
        return x


class _UpsamplingResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, n_layers: int = 3, kernel_size: int = 5,
                 increase_size_factor: int = 2):
        super().__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.increase_size_factor = increase_size_factor

        def build_layer(channels_in, channels_out, kernel_size, decrease_size_factor):
            return nn.Sequential(nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size,
                                                    padding=kernel_size//2, stride=decrease_size_factor),
                                 nn.BatchNorm2d(channels_out),
                                 nn.LeakyReLU())
        layers = [build_layer(self.channels_in, self.channels_out, self.kernel_size, self.increase_size_factor)]
        for i in range(n_layers-1):
            layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, 1))
         # layers.append(build_layer(self.channels_out, self.channels_out, self.kernel_size, self.decrease_size_factor))

        self.layers = nn.ModuleList(layers)

        self.shortcut = nn.ConvTranspose2d(self.channels_in, self.channels_out, kernel_size=1,
                                           stride=self.increase_size_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        x = x + shortcut
        return x


class _ResidualEncoder(nn.Module):
    def __init__(self, bottleneck_size: int, channels: int, initial_channels: int, n_blocks: int, factor: int,
                 dropout: float, height: int, width: int):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channels = channels
        self.initial_channels = initial_channels
        self.n_blocks = n_blocks
        self.factor = factor
        self.dropout = dropout
        self.height = height
        self.width = width

        blocks = [
            _ResidualBlock(channels_in=self.channels, channels_out=self.initial_channels,
                           decrease_size_factor=self.factor)
        ]

        blocks = [_ResidualBlock(channels_in=self.channels, channels_out=self.initial_channels,
                                 decrease_size_factor=self.factor)]
        current_channels = self.initial_channels
        current_height = self.height // factor
        current_width = self.width // factor
        for i in range(self.n_blocks - 1):
            blocks.append(_ResidualBlock(channels_in=current_channels, channels_out=current_channels*self.factor,
                                         decrease_size_factor=self.factor))
            current_channels *= factor
            if current_height % 2 == 0:
                current_height //= factor
            else:
                current_height = current_height//factor + 1
            if current_width % 2 == 0:
                current_width //= factor
            else:
                current_width = current_width // factor + 1
        self.blocks = nn.ModuleList(blocks)

        self.last_size = (current_channels, current_height, current_width)

        def build_linear(in_, out):
            return nn.Sequential(nn.Linear(in_, out), nn.BatchNorm1d(out), nn.LeakyReLU(), nn.Dropout(self.dropout))

        self.fc_mu = build_linear(current_channels*current_height*current_width, self.bottleneck_size)
        self.fc_log_var = build_linear(current_channels*current_height*current_width, self.bottleneck_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class _ResidualDecoder(nn.Module):
    def __init__(self, bottleneck_size: int, initial_channels: int, n_blocks: int, factor: int, initial_height: int,
                 initial_width: int, target_height: int, target_width: int, target_channels: int, dropout: float):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.initial_channels = initial_channels
        self.n_blocks = n_blocks
        self.factor = factor
        self.initial_height = initial_height
        self.initial_width = initial_width
        self.target_height = target_height
        self.target_width = target_width
        self.target_channels = target_channels
        self.dropout = dropout

        self.initial_fc = nn.Sequential(nn.Linear(self.bottleneck_size,
                                                  self.initial_channels*self.initial_width*self.initial_height),
                                                  nn.BatchNorm1d(
                                                      self.initial_channels*self.initial_width*self.initial_height),
                                                  nn.LeakyReLU(), nn.Dropout(self.dropout))

        current_channels = self.initial_channels // factor
        blocks = [_UpsamplingResidualBlock(channels_in=self.initial_channels, channels_out=current_channels,
                                           increase_size_factor=self.factor)]
        current_height = self.initial_height * factor
        current_width = self.initial_width * factor
        for i in range(self.n_blocks-1):
            if i == self.n_blocks-2:
                new_channels = target_channels
            else:
                new_channels = current_channels // factor
            blocks.append(_UpsamplingResidualBlock(channels_in=current_channels, channels_out=new_channels,
                                                   increase_size_factor=self.factor))
            current_channels //= factor
            current_height *= factor
            current_width *= factor
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(x)
        x = x.view(-1, self.initial_channels, self.initial_height, self.initial_width)
        for block in self.blocks:
            x = block(x)
        return x


class ResidualVAE(nn.Module):
    def __init__(self, bottleneck_size: int = 128, channels: int = 3, initial_channels: int = 32, n_blocks: int = 4,
                 factor: int = 2, dropout: float = 0.5, height: int = 218, width: int = 178):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.channels = channels
        self.initial_channels = initial_channels
        self.n_blocks = n_blocks
        self.factor = factor
        self.dropout = dropout
        self.height = height
        self.width = width
        self.encoder = _ResidualEncoder(self.bottleneck_size, self.channels, self.initial_channels, self.n_blocks,
                                        self.factor, self.dropout, self.height, self.width)
        decoder_initial_channels, decoder_initial_height, decoder_initial_width = self.encoder.last_size
        self.decoder = _ResidualDecoder(self.bottleneck_size, decoder_initial_channels, self.n_blocks, self.factor,
                                        decoder_initial_height, decoder_initial_width, height, width, channels,
                                        dropout)

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
