from abc import ABC
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision

from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.utils.klmse_bce import MSEKLDBCELoss
from face_interpolator.utils.unormalize import UnNormalize


class ConditionalAutoEncoderModel(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unorm = UnNormalize(mean=MEAN, std=STD)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)

        # log images
        if batch_idx % 1000 == 0:
            decoded_images = decoded.type_as(x)

            unorm_input = [self.unorm(img) for img in x[:6].detach().clone()]
            unorm_decoded = [self.unorm(img) for img in decoded_images[:6].detach().clone()]

            grid_input = torchvision.utils.make_grid(unorm_input)
            grid_decoded = torchvision.utils.make_grid(unorm_decoded)

            self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
            self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

        loss, MSE, KLD, BCE = MSEKLDBCELoss()(decoded, x, mu, logvar, pred_attr, y)

        batch_dict_output = {
            "loss": loss,
            'MSE': MSE,
            'KLD': KLD,
            'BCE': BCE
        }

        return batch_dict_output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)
        loss, MSE, KLD, BCE = MSEKLDBCELoss()(decoded, x, mu, logvar, pred_attr, y)

        batch_dict_output = {
            "loss": loss,
            'MSE': MSE,
            'KLD': KLD,
            'BCE': BCE
        }

        return batch_dict_output

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)
        loss, MSE, KLD, BCE = MSEKLDBCELoss()(decoded, x, mu, logvar, pred_attr, y)

        batch_dict_output = {
            "loss": loss,
            'MSE': MSE,
            'KLD': KLD,
            'BCE': BCE
        }

        return batch_dict_output

    def training_epoch_end(self, outputs):
        # The function is called after every epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['MSE'] for x in outputs]).mean()
        avg_kld = torch.stack([x['KLD'] for x in outputs]).mean()
        avg_bce = torch.stack([x['BCE'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("MSE/Train", avg_mse, self.current_epoch)
        self.logger.experiment.add_scalar("KLD/Train", avg_kld, self.current_epoch)
        self.logger.experiment.add_scalar("BCE/Train", avg_bce, self.current_epoch)

    def validation_epoch_end(self, outputs):
        # The function is called after every validation epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['MSE'] for x in outputs]).mean()
        avg_kld = torch.stack([x['KLD'] for x in outputs]).mean()
        avg_bce = torch.stack([x['BCE'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Valid", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("MSE/Valid", avg_mse, self.current_epoch)
        self.logger.experiment.add_scalar("KLD/Valid", avg_kld, self.current_epoch)
        self.logger.experiment.add_scalar("BCE/Valid", avg_bce, self.current_epoch)

    def test_epoch_end(self, outputs):
        # The function is called after every test epoch is completed

        # Log average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['MSE'] for x in outputs]).mean()
        avg_kld = torch.stack([x['KLD'] for x in outputs]).mean()
        avg_bce = torch.stack([x['BCE'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("MSE/Test", avg_mse, self.current_epoch)
        self.logger.experiment.add_scalar("KLD/Test", avg_kld, self.current_epoch)
        self.logger.experiment.add_scalar("BCE/Test", avg_bce, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, x: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
