from abc import ABC
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from face_interpolator.utils.klmse import MSEKLDLoss


class ConditionalAutoEncoderModel(pl.LightningModule, ABC):

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar = self(x, y)

        # log images
        if batch_idx % 10 == 0:
            decoded_images = decoded.type_as(x)
            grid_input = torchvision.utils.make_grid(x[:6])
            grid_decoded = torchvision.utils.make_grid(decoded_images[:6])
            self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
            self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)
        # loss = F.mse_loss(decoded, x)

        batch_dict_output = {
            "loss": loss
        }

        return batch_dict_output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x, y)

        loss = MSEKLDLoss()(decoded, x, mu, logvar)
        # loss = F.mse_loss(decoded, x)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x, y)
        loss = F.mse_loss(decoded, x)
        self.log('test_loss', loss)

    def training_epoch_end(self, outputs):
        # The function is called after every epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor, attributes: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, x: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
