import argparse
import pytorch_lightning as pl
from typing import List
from .cnn_vae import ConvVAE
from .conditional_vae import ConditionalConvVAE
from .vq_vae.vq_vae import VQVAE
from .resnet import ResNet


class ModelFactory:
    @classmethod
    def __call__(cls, args: argparse.Namespace) -> pl.LightningModule:
        assert args.model in cls.available_models()
        assert args.attributes_size is not None or args.model not in ['cvae', 'cvqvae', 'resnet']
        if args.model == 'vae':
            return ConvVAE(args.bottleneck)
        if args.model == 'cvae':
            return ConditionalConvVAE(args.bottleneck, args.attributes_size)
        if args.model == 'vqvae':
            return VQVAE()
        if args.model == 'resnet':
            return ResNet(args.attribute_size)
        raise NotImplementedError(args.model)

    @staticmethod
    def available_models() -> List[str]:
        return ['vae', 'cvae', 'vqvae', 'resnet', 'cvqvae']
