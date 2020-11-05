from abc import ABC

import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
import torch

from typing import Any


class AutoEncoderModel(pl.LightningDataModule):

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar = self(x)

        # log images
        if batch_idx % 100 == 0:
            decoded_images = decoded.type_as(x)
            grid_input = torchvision.utils.make_grid(x[:6])
            grid_decoded = torchvision.utils.make_grid(decoded_images[:6])
            self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
            self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

        loss = F.mse_loss(decoded, x)

        batch_dict_output = {
            "loss": loss
        }

        return batch_dict_output

    def training_epoch_end(self, outputs):
        # The function is called after every epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
