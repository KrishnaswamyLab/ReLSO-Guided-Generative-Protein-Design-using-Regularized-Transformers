# imports
import math
import numpy as np

from sklearn.neighbors import kneighbors_graph


import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

import argparse
import wandb

import relso.utils.eval_utils as eval_utils

from relso.nn.transformers import Relso

#############################
# MODELS
##############################


class GradModel_AE(Relso):

    def __init__(self, hparams=None):
        super(GradModel_AE, self).__init__(hparams)

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams

        self.lr = hparams.lr
        self.opt_method = hparams.opt_method


        self.alpha = hparams.alpha_val
        self.gamma = hparams.gamma_val

    def configure_optimizers(self):

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 150], gamma=0.1 )

        return [opt], [lr_sched]


    @property
    def automatic_optimization(self) -> bool:
        return False


    def training_step(self, batch, batch_idx):

        self.g_opt_step += 1

        preds, targets = self.shared_step(batch)

        assert len(preds) == len(targets), f'preds: {len(preds)} targs: {len(targets)}'

        train_loss, train_loss_logs  = self.loss_function(predictions=preds,
                                                        targets=targets,
                                                        alpha=self.alpha,
                                                        gamma=self.gamma)


        train_loss_logs = self.relabel(train_loss_logs, 'train_')

        # result = pl.TrainResult(minimize=train_loss)
        self.log_dict(train_loss_logs, on_step=True, on_epoch=False)

        opt = self.optimizers()

        self.manual_backward(train_loss, opt, retain_graph=True)


        opt.step()
        opt.zero_grad()

        return train_loss

    def validation_step(self, batch, batch_idx):

        preds, targets = self.shared_step(batch)

        
        # only eval on original batch targets
        self.log_accuracy(preds[0][:len(targets[0])], targets[0])

        if len(preds[0]) > 50:
            self.log_smoothness(batch)

        valid_loss, valid_loss_logs =  self.loss_function(preds, targets, self.alpha, self.gamma, True) # valid=True

        valid_loss_logs = self.relabel(valid_loss_logs, 'valid_')

        self.log_dict(valid_loss_logs, on_step=False, on_epoch=True)

        return valid_loss

    # logging functions
    def log_accuracy(self,pred, true):
        acc = (pred.argmax(1) == true)
        acc =  (acc.sum(1).float()/ float(acc.shape[1])).mean().cpu().detach().float()

        self.log("valid acc", acc , on_step=False, on_epoch=True)

    def log_smoothness(self, batch):
        data, *targets = batch
        preds, z_embed = self(data)

        if len(z_embed) > 50:

            # generate graph
            a_matrix = kneighbors_graph(z_embed.cpu().detach(), n_neighbors=5, metric='euclidean', mode='connectivity') 

            # fit smooth
            fit_smooth = eval_utils.get_fit_smoothness(a_matrix, targets[-1].cpu())
            pred_fit_smooth = eval_utils.get_fit_smoothness(a_matrix, preds[1].cpu())

            # seq smooth
            seq_smooth = eval_utils.get_seq_smoothness(a_matrix, targets[0].cpu())
            seq_preds = F.gumbel_softmax(preds[0], tau=1, dim=1, hard=True).transpose(1,2).flatten(1,2)
            pred_seq_smooth = eval_utils.get_seq_smoothness(a_matrix, seq_preds.cpu())

            self.log("valid true fit smooth", fit_smooth ,  on_step=False, on_epoch=True)
            self.log("valid true seq smooth", seq_smooth ,  on_step=False, on_epoch=True)

            self.log("valid pred fit smooth", pred_fit_smooth ,  on_step=False, on_epoch=True)
            self.log("valid pred seq smooth", pred_seq_smooth ,  on_step=False, on_epoch=True)

    


# ---------------------
# Utils
# -------------------------

def str2basemodel(model_name):
    """returns an uninitialized model module

    Args:
        cl_arg ([type]): [description]

    Returns:
        [type]: [description]
    """
    # model dict
    model_dict = {
        'relso': Relso,
                }

    model = model_dict[model_name]

    return model


def str2gradmodel(model_name):

    model = str2basemodel(model_name)

    return model
