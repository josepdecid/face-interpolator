from abc import ABC
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from face_interpolator.utils.constants import MEAN, STD
from face_interpolator.utils.unormalize import UnNormalize


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

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
        decoded, latent_loss = self(x)

        recon_loss = nn.MSELoss()(decoded, x)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        self.log('val_loss', loss)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        decoded, latent_loss = self(x)

        recon_loss = nn.MSELoss()(decoded, x)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

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

    def decode(self, quant_t: torch.Tensor, quant_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
