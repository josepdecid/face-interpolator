from face_interpolator.models.cnn import CNNDecoder
from face_interpolator.models.cnn import CNNEncoder

from face_interpolator.models.model import AutoEncoderModel
from typing import Any
import torch


class ConvVAE(AutoEncoderModel):
    def __init__(self, bottleneck_size: int, channels=3):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = CNNEncoder(bottleneck_size, channels=channels)

        # Decoder
        self.decoder = CNNDecoder(bottleneck_size, channels=channels)

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.type_as(mu)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

    def encode(self, x: torch.Tensor) -> Any:
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

if __name__ == '__main__':
    model = ConvVAE(50)
    data = torch.zeros(2, 3, 218, 178)
    assert model(data)[0].shape == data.shape