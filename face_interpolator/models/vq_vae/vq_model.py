from abc import ABC

import pytorch_lightning as pl
import torchvision
import torch
from torch import nn
from typing import Any

from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.utils.klmse import MSEKLDLoss
from face_interpolator.utils.unormalize import UnNormalize


class VQAutoEncoderModel(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unorm = UnNormalize(mean=MEAN, std=STD)
        self.latent_loss_weight = 0.25

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, latent_loss = self(x)

        recon_loss = nn.MSELoss()(decoded, x)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        # log images
        if batch_idx % 10 == 0:
            decoded_images = decoded.type_as(x)

            unorm_input = [self.unorm(img) for img in x[:6].detach().clone()]
            unorm_decoded = [self.unorm(img) for img in decoded_images[:6].detach().clone()]

            grid_input = torchvision.utils.make_grid(unorm_input)
            grid_decoded = torchvision.utils.make_grid(unorm_decoded)

            self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
            self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, latent_loss = self(x)

        recon_loss = nn.MSELoss()(decoded, x)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, latent_loss = self(x)

        recon_loss = nn.MSELoss()(decoded, x)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # The function is called after every training epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        # The function is called after every validation epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Valid", avg_loss, self.current_epoch)

    def test_epoch_end(self, outputs):
        # The function is called after every test epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, quant_t: torch.Tensor, quant_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
