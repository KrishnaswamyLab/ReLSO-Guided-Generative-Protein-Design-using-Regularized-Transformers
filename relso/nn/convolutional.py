import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb



from relso.nn import bneck
from relso.nn.base import BaseModel, BaseModelVAE, BaseVAEParamModule
from relso.nn.auxnetwork import str2auxnetwork






# ---------------------
# Convolutional Block
# ---------------------

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.

          from:
          https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch
        """
        super(Block, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels))

    def forward(self, x):

        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out

