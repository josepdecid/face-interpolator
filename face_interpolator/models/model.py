from abc import ABC, ABCMeta

import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
import torch

from typing import Any

from face_interpolator.constants import MEAN, STD
from face_interpolator.utils.klmse import MSEKLDLoss
from face_interpolator.utils.unormalize import UnNormalize


class AutoEncoderModel(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unorm = UnNormalize(mean=MEAN, std=STD)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar = self(x)

        # log images
        if batch_idx % 10 == 0:
            decoded_images = decoded.type_as(x)
            grid_input = torchvision.utils.make_grid(x[:6])
            grid_decoded = torchvision.utils.make_grid(decoded_images[:6])

            grid_input = self.unorm(grid_input)
            grid_decoded = self.unorm(grid_decoded)
            self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
            self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

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

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
