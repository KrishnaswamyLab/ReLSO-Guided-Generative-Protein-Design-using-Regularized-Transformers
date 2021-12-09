import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb


#####################
# Bottlenecks
#####################


class BaseBottleneck(nn.Module):
    """Basic fcn bottleneck

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, h):
        """
        b = batch size
        h = hidden dimension
        z = latent dim
        
        input_dim: b x s x h
        output_dim: b x z
        """

        z_rep = self.fc1(h)

        return z_rep


