
import os
import numpy as np
import argparse
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from phate import PHATE


import wandb
from pytorch_lightning.loggers import WandbLogger
import datetime


import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import relso.grad.models as hmodels
import relso.data as hdata

from relso.optim import utils, optim_algs
from relso.utils import eval_utils


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)
    parser.add_argument('--project_name', default='relso-eval', type=str)
    parser.add_argument('--alpha', required=False, type=float)
    parser.add_argument('--e2e_train', default=False, type=str2bool)


    cl_args = parser.parse_args()

    # logging
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f'end2end_logs/{cl_args.model}_{cl_args.dataset}_{cl_args.alpha}'  
    if cl_args.e2e_train:
        save_dir += '_e2e/'
    else:
        save_dir += '/'

    wandb_logger = WandbLogger(name=f'run_{cl_args.model}_{cl_args.dataset}',
                                project=cl_args.project_name,
                                log_model=False,
                                save_dir=save_dir,
                                offline=False)

    wandb_logger.log_hyperparams(cl_args.__dict__)
    wandb_logger.experiment.log({"logging timestamp":date_suffix})

    # load model
    model = utils.import_model_from_ckpt(cl_args.model, cl_args.weights)
    model.eval()

    wandb_logger.experiment.log({"alpha_val":model.alpha_val})
    wandb_logger.experiment.log({"latent_dim":model.latent_dim})
    wandb_logger.experiment.log({"hidden_dim":model.hidden_dim})

    # load dataset
    proto_data = hdata.str2data(cl_args.dataset)
    data = proto_data(dataset=cl_args.dataset,
                      task='recon',
                      batch_size=100)

    model.seq_len = data.seq_len
    *_, train_targs = data.train_split.tensors
    train_targs = train_targs.numpy()


    # if alpha_val = 0, train prediction head
    if model.alpha_val == 0:
        model, trainer = utils.train_prediction_head(model=model,
                                            data=data,
                                            wandb_logger=wandb_logger,
                                            e2e_train=cl_args.e2e_train)

        trainer.save_checkpoint(save_dir + 'model_state.ckpt')

    # get model evaluations
    # get prediction evaluations
    
    print("performing model evaluations")

    train_seq, _, train_targs = data.train_split.tensors
    valid_seq, _, valid_targs = data.valid_split.tensors
    test_seq, _, test_targs = data.test_split.tensors

    train_n = train_seq.shape[0]
    valid_n = valid_seq.shape[0]
    test_n = test_seq.shape[0]

    print("performing model forward pass")
    with torch.no_grad():
        train_outputs, train_embed = model(train_seq)
        valid_outputs, valid_embed = model(valid_seq)
        test_outputs, test_embed = model(test_seq)

    print("finished model forward pass")

    train_embed = train_embed.reshape(train_n, -1).numpy()
    valid_embed = valid_embed.reshape(valid_n, -1).numpy()
    test_embed = test_embed.reshape(test_n, -1).numpy()

    targets_list = [train_targs, valid_targs, test_targs]
    recon_targ_list = [train_seq, valid_seq, test_seq]
    embed_list = [train_embed, valid_embed, test_embed]

    predictions_list = [x[1] for x in [train_outputs, valid_outputs, test_outputs]]
    recon_list = [x[0] for x in [train_outputs, valid_outputs, test_outputs]]

    seqd_list = [data.train_split_seqd, data.valid_split_seqd, data.test_split_seqd]


    eval_utils.get_all_smoothness_values(targets_list=targets_list,
                                        seqd_list = seqd_list,
                                        embeddings_list=embed_list,
                                        wandb_logger=wandb_logger)


    eval_utils.get_all_fitness_pred_metrics(targets_list=targets_list,
                                            predictions_list=predictions_list,
                                            wandb_logger=wandb_logger)

    eval_utils.get_all_recon_pred_metrics(targets_list=recon_targ_list,
                                        predictions_list=recon_list,
                                        wandb_logger=wandb_logger)


    