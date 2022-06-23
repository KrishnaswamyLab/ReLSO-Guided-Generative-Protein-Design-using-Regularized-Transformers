import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb


import relso.utils.eval_utils as eval_utils

# -------------------------
# BASE MODEL
# -------------------------
class BaseModel(LightningModule):
    def __init__(self, hparams=None):
        super(BaseModel, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        # self.save_hyperparameters()

        # self.hparams = hparams

        self.lr = hparams.lr

    def configure_optimizers(self):
        opt = (torch.optim.Adam(self.parameters(), lr=self.lr),)
        return opt

    def shared_step(self, batch):

        # batch is x_data, x_target, y_target
        data, *targets = batch

        preds, _ = self(data)

        return preds, targets

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):

        preds, targets = self.shared_step(batch)

        assert len(preds) == len(targets), f"preds: {len(preds)} targs: {len(targets)}"

        train_loss, train_loss_logs = self.loss_function(
            predictions=preds, targets=targets
        )

        train_loss_logs = self.relabel(train_loss_logs, "train_")

        # result = pl.TrainResult(minimize=train_loss)
        self.log_dict(train_loss_logs, on_step=True, on_epoch=False)

        return train_loss

    def loss_function(self, predictions, targets, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
            loss dict
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError


class BaseModelVAE(LightningModule):
    def __init__(self, hparams=None):
        super(BaseModelVAE, self).__init__()

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters()

        self.hparams = hparams
        self.lr = self.hparams.lr

        self.mu = None
        self.logvar = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def shared_step(self, batch):

        # batch is x_data, x_target, y_target
        data, *targets = batch

        preds, _ = self(data)

        return preds, targets

    def relabel(self, loss_dict, label):

        loss_dict = {label + str(key): val for key, val in loss_dict.items()}

        return loss_dict

    def training_step(self, batch, batch_idx):

        preds, targets = self.shared_step(batch)

        assert len(preds) == len(targets), f"preds: {len(preds)} targs: {len(targets)}"

        train_loss, train_loss_logs = self.loss_function(
            predictions=preds, targets=targets, mu=self.mu, logvar=self.logvar
        )

        train_loss_logs = self.relabel(train_loss_logs, "train_")

        # result = pl.TrainResult(minimize=train_loss)
        self.log_dict(train_loss_logs, on_step=True, on_epoch=False)

        return train_loss

    def loss_function(self, predictions, targets, mu, logvar, valid_step=False):
        """
        takes in predictions and targets

        predictions and targets can be a list

        Args:
            predictions (dict): [description]
            targets (dict): [description]

        Returns:
            loss value
            loss dict
        """
        raise NotImplementedError

    def forward(self, batch):
        """
        the forward method should just take data
        as an argument


        should return:
            -  predictions
            -  embeddings
        """
        return NotImplementedError


class BaseVAEParamModule(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseVAEParamModule, self).__init__()

        self.mu_layer = nn.Linear(input_dim, bottleneck_dim)
        self.var_layer = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):

        mu = self.mu_layer(h)
        var = self.var_layer(h)

        return mu, var
