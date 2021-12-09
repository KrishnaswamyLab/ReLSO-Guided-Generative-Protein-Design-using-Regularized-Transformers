import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


class BaseRegressor(nn.Module):
    def __init__(self, hparams):
        super(BaseRegressor, self).__init__()

        latent_dim = hparams.latent_dim

        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn_reg = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)
        self.nonlin = nn.ReLU()

    def forward(self, z):

        h =  self.nonlin(self.bn_reg(self.fc1(z)))
        y_hat = self.fc2(h)
        return y_hat


class DropoutRegressor(nn.Module):
    def __init__(self, hparams):
        super(DropoutRegressor, self).__init__()

        latent_dim = hparams.latent_dim
        drop_prob = hparams.probs

        self.fc1 = nn.Linear(latent_dim, 64)
        self.dropout = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z):

        h = self.dropout(F.relu(self.fc1(z)))
        y_hat = self.fc2(h)
        return y_hat


def str2auxnetwork(auxnetwork_name):
    """returns an uninitialized model module

    Args:
        cl_arg ([type]): [description]

    Returns:
        [type]: [description]
    """
    # model dict
    auxnetwork_dict = {'base_reg': BaseRegressor,
                'dropout_reg': DropoutRegressor}

    auxnetwork = auxnetwork_dict[auxnetwork_name]

    return auxnetwork
