from abc import ABC

import pytorch_lightning as pl
import torchvision
import torch

from typing import Any

from face_interpolator.utils.constants import MEAN, STD
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

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

        # log images
        if batch_idx % 1000 == 0:
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
        decoded, mu, logvar = self(x)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)

        return {"test_loss": loss}

    def training_epoch_end(self, outputs):
        # The function is called after every training epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        # The function is called after every validation epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Valid", avg_loss, self.current_epoch)

    def test_epoch_end(self, outputs):
        # The function is called after every test epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
