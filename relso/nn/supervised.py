
import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb

from relso.nn.auxnetwork import str2auxnetwork
from relso.nn import bneck


# LSTM model

class LSTM(BaseModel):
    def __init__(self, hparams):
        super(LSTM, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.embedding_dim = hparams.embedding_dim
        self.hidden_dim = hparams.hidden_dim
        self.layers = hparams.layers
        self.probs = hparams.probs
        self.bidirectional = hparams.bidirectional

        # Embedding layer
        self.cdr_embedding = nn.Embedding(self.input_dim,self.embedding_dim)

        if self.bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1

        # LSTM layer
        if self.bidirectional == True:
            self.lstm = nn.LSTM(self.embedding_dim,
                                self.hidden_dim,
                                num_layers = self.layers,
                                batch_first=True,
                                dropout = self.probs,
                                bidirectional = True)

        else:
            self.lstm = nn.LSTM(self.embedding_dim,
                                self.hidden_dim,
                                num_layers = self.layers,
                                dropout = self.probs)

        self.bottleneck_module = bneck.BaseBottleneck(self.num_dir*self.layers*self.hidden_dim, self.latent_dim) 

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            self.regressor_module = auxnetwork(hparams)
        except:
            auxnetwork = str2auxnetwork('base_reg')
            self.regressor_module = auxnetwork(hparams)

    def encoder(self, batch):

        embedded_batch = self.cdr_embedding(batch)

        rep,(h_n, _) = self.lstm(embedded_batch)

        # h_n: torch.Size([4, 100, 50])
        h_rep = h_n.transpose(0,1).reshape(-1, self.num_dir*self.layers*self.hidden_dim)

        # pass through bottleneck
        z_rep = self.bottleneck_module(h_rep)

        self.z_rep = z_rep

        return z_rep



    def forward(self, batch):

        z_rep = self.encoder(batch)

        y_hat = self.regressor_module(z_rep)

        x_hat = torch.Tensor([0])

        return [x_hat, y_hat], z_rep

    def loss_function(self, predictions, targets, valid_step=True):

        # unpack tensors
        _, y_hat = predictions
        _, y_t = targets

        loss = nn.MSELoss()(y_hat.flatten(), y_t.flatten())

        return loss, {'loss': loss}


    def validation_step(self, batch, batch_idx):

        preds, targets = self.shared_step(batch)

        if len(preds[0]) > 50:
            self.log_smoothness(batch)

        valid_loss, valid_loss_logs =  self.loss_function(preds, targets, True) # valid=True

        valid_loss_logs = self.relabel(valid_loss_logs, 'valid_')

        # result = pl.EvalResult(checkpoint_on=valid_loss, early_stop_on=valid_loss)
        self.log_dict(valid_loss_logs, on_step=False, on_epoch=True)

        return valid_loss



# ----------------------
# CNN Model
# ----------------------


class CNN(BaseModel):
    def __init__(self, hparams):
        super(CNN, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams
        self.input_dim = hparams.input_dim
        self.latent_dim = hparams.latent_dim
        self.hidden_dim = hparams.hidden_dim
        self.kernel_size = hparams.kernel_size
        self.layers = hparams.layers
        self.dropout = hparams.probs
        # self.alpha_val = hparams.alpha_val


        self.conv1 = nn.Conv1d(in_channels=self.input_dim,
                               out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              )

        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              )

        self.pool = nn.MaxPool1d(1)
        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

        self.len_out = self.calc_len_out(hparams.layers,
                                    hparams.hidden_dim,
                                    hparams.seq_len,
                                    hparams.kernel_size)

        # auxiliary network
        self.bottleneck_module = BaseBottleneck(self.len_out*self.hidden_dim,self.latent_dim)

        try:
            auxnetwork = str2auxnetwork(hparams.auxnetwork)
            self.regressor_module = auxnetwork(hparams)
        except:
            auxnetwork = str2auxnetwork('base_reg')
            self.regressor_module = auxnetwork(hparams)


    def one_hot_embedding(self,labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
        labels: (LongTensor) class labels, sized [N,].
        num_classes: (int) number of classes.

        Returns:
        (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes,  device=self.device)

        return y[labels]

    def encoder(self, batch):

        # convert to one-hot
        # One hot encoding buffer that you create out of the loop and just keep reusing
        x_onehot = self.one_hot_embedding(batch, self.hparams.input_dim).transpose(1,2)

        h = self.pool(F.relu(self.conv1(x_onehot)))
        h = self.pool(F.relu(self.conv2(h)))

        # flatten
        _, f_dim, f_len = h.shape
        h_rep = h.view(-1, f_dim*f_len)
        z_rep = self.bottleneck_module(h_rep)


        self.z_rep = z_rep

        return z_rep


    # NEW FUNCTION
    def forward(self, batch):

        z_rep = self.encoder(batch)

        y_hat = self.regressor_module(z_rep)

        x_hat = torch.Tensor([0])

        return [x_hat, y_hat], z_rep

    def loss_function(self, predictions, targets, valid_step=False):

        _, y_hat = predictions
        _, y_t = targets

        loss = nn.MSELoss()(y_hat.flatten(), y_t.flatten())

        return loss, {'loss': loss}

    @staticmethod
    def calc_len_out(depth, hidden_dim, input_len, kernel_size, stride=1, padding=0, dilation=1):
        """
        L_out = floor( (L_in  + 2*padding - dilation*(kernel size - 1)-1)/stride + 1)
        from: https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
        """

        flat_len = input_len

        print("getting flat len")
        for _ in range(depth):
            print("flat len: {}".format(flat_len))
            flat_len = flat_len  + 2*padding - dilation * (kernel_size - 1) - 1
            flat_len = int(flat_len/stride) # int takes floor
            flat_len += 1

        return flat_len


    def validation_step(self, batch, batch_idx):

        preds, targets = self.shared_step(batch)

        if len(preds[0]) > 50:
            self.log_smoothness(batch)

        valid_loss, valid_loss_logs =  self.loss_function(preds, targets, True) # valid=True

        valid_loss_logs = self.relabel(valid_loss_logs, 'valid_')

        # result = pl.EvalResult(checkpoint_on=valid_loss, early_stop_on=valid_loss)
        self.log_dict(valid_loss_logs, on_step=False, on_epoch=True)

        return valid_loss

