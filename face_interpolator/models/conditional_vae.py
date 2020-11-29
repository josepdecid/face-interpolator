from typing import Any

import torch

from models.conditional_cnn import ConditionalEncoder, ConditionalDecoder
from models.conditional_model import ConditionalAutoEncoderModel


class ConditionalConvVAE(ConditionalAutoEncoderModel):
    def __init__(self, bottleneck_size: int, attribute_size, channels=3):
        super(ConditionalConvVAE, self).__init__()

        # Encoder
        self.encoder = ConditionalEncoder(bottleneck_size, attribute_size, channels=channels)

        # Decoder
        self.decoder = ConditionalDecoder(bottleneck_size, attribute_size, channels=channels)

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.type_as(mu)

        return eps.mul(std).add_(mu)

    def forward(self, x, attributes):
        mu, logvar = self.encode(x, attributes)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z, attributes)
        return decoded, mu, logvar

    def encode(self, x: torch.Tensor, attributes: torch.Tensor) -> Any:
        mu, logvar = self.encoder(x, attributes)
        return mu, logvar

    def decode(self, x: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, attributes)


if __name__ == '__main__':
    attribute_size = 20
    model = ConditionalConvVAE(50, attribute_size)
    data = torch.zeros(2, 3, 218, 178)
    attributes = torch.zeros(2, attribute_size)
    assert model(data, attributes)[0].shape == data.shape
