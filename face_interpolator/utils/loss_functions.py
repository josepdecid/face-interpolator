import torch
from torch import nn


class _KLD(nn.Module):
    def forward(self, log_var, mu):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


class AutoEncoderLoss(nn.Module):

    def __init__(self, lambda_mse: int = 1, lambda_kld: int = 1, lambda_bce: int = 0):
        super().__init__()
        assert lambda_mse + lambda_kld + lambda_bce > 0
        self.lambda_kld = lambda_kld
        self.lambda_mse = lambda_mse
        self.lambda_bce = lambda_bce
        if lambda_mse > 0:
            self.mse_loss = nn.MSELoss(reduction='sum')
        if lambda_bce > 0:
            self.bce_loss = nn.BCELoss(reduction='sum')
        if lambda_kld > 0:
            self.kld = _KLD()

    def forward(self, x1, x2, mu, log_var, pred_att=None, att=None):
        mse = self.mse_loss(x1, x2) * self.lambda_mse if self.lambda_mse > 0 else 0
        bce = self.bce_loss(pred_att, att) * self.lambda_bce if self.lambda_bce > 0 else 0
        kld = self.kld(log_var, mu) * self.lambda_kld if self.lambda_kld > 0 else 0
        total = mse + bce + kld
        return total, dict(loss=total.item(), MSE=mse, BCE=bce, KLD=kld)

