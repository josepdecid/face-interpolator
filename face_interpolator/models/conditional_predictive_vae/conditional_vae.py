from typing import Any

import torch

from models.conditional_predictive_vae.conditional_cnn import ConditionalEncoder, ConditionalDecoder
from models.conditional_predictive_vae.conditional_model import ConditionalAutoEncoderModel


class ConditionalConvVAE(ConditionalAutoEncoderModel):
    def __init__(self, bottleneck_size: int, attribute_size, channels=3):
        super(ConditionalConvVAE, self).__init__()

        # Encoder
        self.encoder = ConditionalEncoder(bottleneck_size, channels=channels)

        # Attr
        self.attribute_predicter = torch.nn.Linear(bottleneck_size, attribute_size)
        self.sigmoid = torch.nn.Sigmoid()
        # Decoder
        self.decoder = ConditionalDecoder(bottleneck_size, attribute_size, channels=channels)

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size()).normal_()
        eps = eps.type_as(mu)

        return eps.mul(std).add_(mu)

    def forward(self, x, attr):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pred_attr = self.predict_attributes(z)
        if self.training:
            decoded = self.decode(z, attr)
        else:
            decoded = self.decode(z, pred_attr)

        return decoded, mu, logvar, pred_attr

    def encode(self, x: torch.Tensor) -> Any:
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, x: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, attributes)

    def predict_attributes(self, z):
        return self.sigmoid(self.attribute_predicter(z))


if __name__ == '__main__':
    attribute_size = 20
    model = ConditionalConvVAE(50, attribute_size)
    data = torch.zeros(2, 3, 218, 178)
    attributes = torch.zeros(2, attribute_size)
    assert model(data, attributes)[0].shape == data.shape
    assert model(data, attributes)[-1].shape == attributes.shape