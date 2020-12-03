import torch
from torch import nn


class MSEKLDBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.lambda_mse = 1
        self.lambda_kld = 1
        self.lambda_bce = 1

    def forward(self, x1, x2, mu, log_var, pred_att, att):
        MSE = self.mse_loss(x1, x2) * self.lambda_mse
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.lambda_kld
        BCE = self.bce_loss(pred_att, att) * self.lambda_bce
        return MSE + KLD + BCE, MSE, KLD, BCE
