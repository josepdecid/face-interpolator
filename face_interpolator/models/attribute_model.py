from abc import ABC
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn


class AttributeModel(pl.LightningModule, ABC):

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        output = self(x)

        loss = nn.MSELoss()(output, y)

        batch_dict_output = {
            "loss": loss
        }

        return batch_dict_output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        loss = nn.MSELoss()(output, y)

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        loss = nn.MSELoss()(output, y)
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
