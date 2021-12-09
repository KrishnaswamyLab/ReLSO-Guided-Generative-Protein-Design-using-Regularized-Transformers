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
# Pooling
#####################

class AttnPooling(nn.Module):
    """Pools by attention weights

    a wieghted sum where weights are learned and sum to 1

    Args:
        nn ([type]): [description]
    """
    def __init__(self, input_dim, reduce_first=False):
        super(AttnPooling, self).__init__()


        self.reduce_first = reduce_first

        if reduce_first:
            self.fc1 = nn.Linear(input_dim, input_dim//2)

        self.glob_attn_module = nn.Sequential(nn.Linear(input_dim, 1),
                                            nn.Softmax(dim=2))

    def forward(self, h):
        """
        b = batch size
        s = sequence_length
        i = input dimension

        input_dim: b x s x i
        output_dim: b x i

        """

        if self.reduce_first:
            h = self.fc1(h)
   
        glob_attn = self.glob_attn_module(h)

        out = torch.bmm(glob_attn.transpose(-1, 1), h).squeeze()

        if h.shape[0] == 1:
            # restore batch dimension
            out = out.unsqueeze(0)

        return out

