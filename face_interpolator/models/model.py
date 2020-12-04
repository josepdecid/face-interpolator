from abc import ABC

import pytorch_lightning as pl
import torchvision
import torch

from typing import Any, Optional

from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.utils.loss_functions import AutoEncoderLoss
from face_interpolator.utils.unormalize import UnNormalize


class AutoEncoderModel(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lambda_bce=0, **kwargs)
        self.unorm = UnNormalize(mean=MEAN, std=STD)
        self.loss = AutoEncoderLoss(lambda_bce=0)

    def _log_images(self, x, decoded):
        decoded_images = decoded.type_as(x)

        unorm_input = [self.unorm(img) for img in x[:6].detach().clone()]
        unorm_decoded = [self.unorm(img) for img in decoded_images[:6].detach().clone()]

        grid_input = torchvision.utils.make_grid(unorm_input)
        grid_decoded = torchvision.utils.make_grid(unorm_decoded)

        self.logger.experiment.add_image('Input Images', grid_input, self.current_epoch)
        self.logger.experiment.add_image('Generated Images', grid_decoded, self.current_epoch)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = self.loss(decoded, x, mu, logvar)

        # log images
        if batch_idx % 1000 == 0:
            self._log_images(self, x, decoded)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = self.loss(decoded, x, mu, logvar)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar = self(x)

        loss = self.loss(decoded, x, mu, logvar)

        return {"test_loss": loss}

    def _aggregate_log(self, outputs, key, subset, name=None):
        if name is None:
            name = key
        if key not in outputs[0]:
            return
        avg = torch.stack([x[key] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f'{name}/{subset}', avg, self.current_epoch)

    def _aggregate_logs(self, outputs, subset):
        assert subset in ['Train', 'Valid', 'Test']
        loss_name = 'Loss'
        if subset == 'Train':
            loss_key = 'loss'
        elif subset == 'Valid':
            loss_key = 'val_loss'
        else:
            loss_key = 'test_loss'
        self._aggregate_log(outputs, loss_key, subset, name=loss_name)
        self._aggregate_log(outputs, 'MSE', subset)
        self._aggregate_log(outputs, 'BCE', subset)
        self._aggregate_log(outputs, 'KLD', subset)

    def training_epoch_end(self, outputs):
        # The function is called after every epoch is completed
        # Log average loss
        self._aggregate_logs(outputs, 'Train')

    def validation_epoch_end(self, outputs):
        # The function is called after every validation epoch is completed
        # Log average loss
        self._aggregate_logs(outputs, 'Valid')

    def test_epoch_end(self, outputs):
        # The function is called after every test epoch is completed
        # Log average loss
        self._aggregate_logs(outputs, 'Test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, x: torch.Tensor) -> Any:
        raise NotImplementedError()

    def decode(self, x: torch.Tensor, attributes: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class ConditionalAutoEncoderModel(AutoEncoderModel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, lambda_bce=1, **kwargs)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)

        # log images
        if batch_idx % 1000 == 0:
            self._log_images(x, decoded)

        loss, loss_dict = self.loss(decoded, x, mu, logvar, pred_attr, y)

        return loss_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)
        loss, loss_dict = self.loss(decoded, x, mu, logvar, pred_attr, y)

        loss_dict['val_loss'] = loss_dict['loss']
        del loss_dict['loss']

        return loss_dict

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, mu, logvar, pred_attr = self(x)
        loss, loss_dict = self.loss(decoded, x, mu, logvar, pred_attr, y)

        loss_dict['test_loss'] = loss_dict['loss']
        del loss_dict['loss']

        return loss_dict