import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data

from models.cnn_decoder import CNNDecoder
from models.cnn_encoder import CNNEncoder


class ConvVAE(pl.LightningModule):
    def __init__(self, bottleneck_size, channels=3):
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
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar = self(x)
        loss = F.mse_loss(decoded, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
