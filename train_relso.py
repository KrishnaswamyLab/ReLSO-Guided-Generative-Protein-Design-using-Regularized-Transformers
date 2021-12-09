
import datetime
import time
import os
import numpy as np
import argparse
from argparse import ArgumentParser
from phate import PHATE
from sklearn.decomposition import PCA
# from sklearn.metrics import r2_score
from scipy.stats import pearsonr as r2_score
import matplotlib.pyplot as plt
import wandb

import torch
from torch import nn, optim

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule

import relso.grad.models as hmodels
import relso.data as hdata
from relso.utils import data_utils, eval_utils


########################
# CONSTANTS
########################

 
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

    tic = time.perf_counter()

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument('--dataset', required=True, type=str)

    # data argmuments
    parser.add_argument('--input_dim', default=22, type=int)
    parser.add_argument('--task', default='recon', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--project_name', default='relso_project', type=str)

    # training arguments
    parser.add_argument('--alpha_val', default=1.0, type=float)
    parser.add_argument('--beta_val', default=0.0005, type=float)
    parser.add_argument('--gamma_val', default=1.0, type=float )
    parser.add_argument('--sigma_val', default=1.5, type=float)

    parser.add_argument('--eta_val', default=0.001, type=float)

    parser.add_argument('--reg_ramp', default=False, type=str2bool)
    parser.add_argument('--vae_ramp', default=True, type=str2bool)

    parser.add_argument('--neg_samp', default=True, type=str2bool)
    parser.add_argument('--neg_size', default=16, type=int)
    parser.add_argument('--neg_weight', default=0.8, type=float)
    parser.add_argument('--neg_floor', default=-2.0, type=float)
    parser.add_argument('--neg_norm', default=4.0, type=float)
    parser.add_argument('--neg_focus', default=False, type=str2bool)

    parser.add_argument('--interp_samp', default=True, type=str2bool)
    parser.add_argument('--interp_size', default=16, type=int)
    parser.add_argument('--interp_weight', default=0.001, type=float)

    parser.add_argument('--wl2norm', default=False, type=str2bool)
    

    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--n_epochs', default=400, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--dev', default=False, type=str2bool)
    parser.add_argument('--seq_len', default=0, type=int)
    parser.add_argument('--auto_lr', default=False, type=str2bool)
    parser.add_argument('--seqdist_cutoff', default=None)

    # LSTM
    parser.add_argument('--embedding_dim', default=100, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool)

    # CNN
    parser.add_argument('--kernel_size', default=4, type=int)

    # BOTH
    parser.add_argument('--latent_dim', default=30, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--probs', default=0.2, type=float)
    parser.add_argument('--auxnetwork', default='base_reg', type=str)



    # ---------------------------
    # CLI ARGS
    # ---------------------------
    cl_args = parser.parse_args()

    print("now training")
    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # ---------------------------
    # LOGGING
    # ---------------------------
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")

    if cl_args.log_dir and cl_args.log_dir == 'e2e':
        save_dir = f'end2end_logs/{cl_args.model}_{cl_args.dataset}_{cl_args.alpha_val}/'

    elif cl_args.log_dir:
        save_dir = cl_args.log_dir + f'{cl_args.model}/{cl_args.dataset}/{date_suffix}/'

    else:
        save_dir = f'train_logs/{cl_args.model}/{cl_args.dataset}/{date_suffix}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    wandb_logger = WandbLogger(name=f'run_{cl_args.model}_{cl_args.dataset}',
                                project=cl_args.project_name,
                                log_model=True,
                                save_dir=save_dir,
                                offline=False)

    wandb_logger.log_hyperparams(cl_args.__dict__)
    wandb_logger.experiment.log({"logging timestamp":date_suffix})

    # ---------------------------
    # TRAINING
    # ---------------------------
    early_stop_callback = EarlyStopping(
            monitor='valid fit smooth', # set in EvalResult
            min_delta=0.001,
            patience=8,
            verbose=True,
            mode='min'
            )

    # get models

    proto_data = hdata.str2data(cl_args.dataset)


    proto_model = hmodels.str2gradmodel(cl_args.model)


    # initialize both model and data
    data = proto_data(dataset=cl_args.dataset,
                task=cl_args.task,
                batch_size=cl_args.batch_size,
                seqdist_cutoff=cl_args.seqdist_cutoff)

    cl_args.seq_len = data.seq_len

    model = proto_model(hparams=cl_args)

    trainer = pl.Trainer.from_argparse_args(cl_args,
                                            max_epochs=cl_args.n_epochs,
                                            max_steps=300000,
                                            gpus='1',
                                            #callbacks=[early_stop_callback],
                                            logger=wandb_logger,
                                            fast_dev_run=cl_args.dev,
                                            gradient_clip_val=1,
                                            auto_lr_find=cl_args.auto_lr,
                                            automatic_optimization= False)
                                            #automatic_optimization= not cl_args.track_grads)

    # Run learning rate finder if selected
    if cl_args.auto_lr:
        print("auto learning rate enabled")
        print("selecting optimal learning rate")
        lr_finder = trainer.tuner.lr_find(model,
                                        train_dataloader=data.train_dataloader())

        #pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("old lr: {} | new lr: {}".format(cl_args.lr, new_lr))

        # update hparams of the model
        model.hparams.lr = new_lr
        wandb_logger.experiment.log({"auto_find_lr": new_lr})


    trainer.fit(model=model,
                train_dataloader=data.train_dataloader(),
                val_dataloaders=data.valid_dataloader(),
                )

    # save model
    trainer.save_checkpoint(save_dir + 'model_state.ckpt')

    model.eval()
    model.cpu()

    print('\ntraining complete!\n')
    # ---------------------
    # EVALUATION
    # ---------------------

    print("now beginning evaluations...\n")
    # Load raw data using load_rawdata, which gives indices + enrichment

    train_reps, _ , train_targs = data.train_split.tensors # subset objects
    valid_reps, _ , valid_targs = data.valid_split.tensors # subset objects
    test_reps, _ , test_targs = data.test_split.tensors

    print("train sequences raw shape: {}".format(train_reps.shape))
    print("valid sequences raw shape: {}".format(valid_reps.shape))
    print("test sequences raw shape: {}".format(test_reps.shape))

    np.save(save_dir+ "train_fitness_array.npy", train_targs.numpy())
    np.save(save_dir+ "valid_fitness_array.npy", valid_targs.numpy())
    np.save(save_dir+ "test_fitness_array.npy", test_targs.numpy())

    train_n = train_reps.shape[0]
    valid_n = valid_reps.shape[0]
    test_n = test_reps.shape[0]

    print('getting embeddings')
    train_outputs, train_hrep = eval_utils.get_model_outputs(model, train_reps)
    valid_outputs, valid_hrep = eval_utils.get_model_outputs(model, valid_reps)
    test_outputs, test_hrep = eval_utils.get_model_outputs(model, test_reps)
    

    print("model has fitness predictions")
    train_recon, train_fit_pred = train_outputs
    valid_recon, valid_fit_pred = valid_outputs
    test_recon, test_fit_pred = test_outputs

    print("shape of train outputs:")
    print(f'{train_recon.shape}, {train_fit_pred.shape}')


    targets_list = [train_targs, valid_targs, test_targs]
    recon_targ_list = [train_reps, valid_reps, test_reps]

    predictions_list = [x[1] for x in [train_outputs, valid_outputs, test_outputs]]
    recon_list = [x[0] for x in [train_outputs, valid_outputs, test_outputs]]
    seqd_list = [data.train_split_seqd, data.valid_split_seqd, data.test_split_seqd]



    # ------------------------------------------------
    # EMBEDDING EVALUATION
    # ------------------------------------------------
    print("saving embeddings")

    train_embed = train_hrep.reshape(train_n, -1).numpy()
    valid_embed = valid_hrep.reshape(valid_n, -1).numpy()
    test_embed = test_hrep.reshape(test_n, -1).numpy()

    embed_list = [train_embed, valid_embed, test_embed]

    print("train embedding shape: {}".format(train_embed.shape))
    print("valid embedding shape: {}".format(valid_embed.shape))
    print("test embedding shape: {}".format(test_embed.shape))

    np.save(save_dir+ "train_embeddings.npy", train_embed)
    np.save(save_dir+ "valid_embeddings.npy", valid_embed)
    np.save(save_dir+ "test_embeddings.npy", test_embed)

    print("logging plots")

    # ---------------------
    # PCA
    # ---------------------
    print("running PCA on embeddings")
    train_pca_coords = PCA(n_components=2).fit_transform(train_embed)
    valid_pca_coords = PCA(n_components=2).fit_transform(valid_embed)
    test_pca_coords =  PCA(n_components=2).fit_transform(test_embed)

    # PCA plot - FITNESS VALUES
    fig, ax = plt.subplots(1,3, figsize=(30,10))


    # TRAIN SET
    im1 = ax[0].scatter(train_pca_coords[:,0],
                        train_pca_coords[:,1],
                        c=train_targs.numpy(),
                        s=5)

    ax[0].title.set_text("train set embedding PCA")
    plt.colorbar(im1, ax=ax[0])

    # VALID SET
    im2 = ax[1].scatter(valid_pca_coords[:,0],
                        valid_pca_coords[:,1],
                        c=valid_targs.numpy(),
                        s=5)

    ax[1].title.set_text("valid set embedding PCA")
    plt.colorbar(im2, ax=ax[1])


    # TEST SET
    im3 = ax[2].scatter(test_pca_coords[:,0],
                        test_pca_coords[:,1],
                        c=test_targs.numpy(),
                        s=10)

    ax[2].title.set_text("test set embedding PCA")
    plt.colorbar(im3, ax=ax[2])

    wandb_logger.experiment.log({"latent space PCA w fitness values": wandb.Image(plt)})


     # PCA plot - Sequence Distance Values
    fig, ax = plt.subplots(1,3, figsize=(30,10))


    # TRAIN SET
    im1 = ax[0].scatter(train_pca_coords[:,0],
                        train_pca_coords[:,1],
                        c=data.train_split_seqd,
                        s=5)

    ax[0].title.set_text("train set embedding PCA")
    plt.colorbar(im1, ax=ax[0])

    # VALID SET
    im2 = ax[1].scatter(valid_pca_coords[:,0],
                        valid_pca_coords[:,1],
                        c=data.valid_split_seqd,
                        s=5)

    ax[1].title.set_text("valid set embedding PCA")
    plt.colorbar(im2, ax=ax[1])


    # TEST SET
    im3 = ax[2].scatter(test_pca_coords[:,0],
                        test_pca_coords[:,1],
                        c=data.test_split_seqd,
                        s=10)

    ax[2].title.set_text("test set embedding PCA")
    plt.colorbar(im3, ax=ax[2])

    wandb_logger.experiment.log({"latent space PCA w sequence distances": wandb.Image(plt)})




    # ---------------------
    # PHATE
    # ---------------------

    try:
        train_embed_phate = PHATE(n_components=2).fit_transform(train_embed)
        valid_embed_phate = PHATE(n_components=2).fit_transform(valid_embed)
        test_embed_phate = PHATE(n_components=2).fit_transform(test_embed)

        np.save(save_dir+ "train_phate_coords.npy", train_embed_phate)
        np.save(save_dir+ "valid_phate_coords.npy", valid_embed_phate)
        np.save(save_dir+ "test_phate_coords.npy", test_embed_phate)

        # PHATE plot - FITNESS VALUES
        # -------------------------------------
        fig, ax = plt.subplots(1,3, figsize=(30,10))

        # TRAIN SET
        im1 = ax[0].scatter(train_embed_phate[:,0],
                            train_embed_phate[:,1],
                            c=train_targs.numpy(),
                            s=5)

        ax[0].title.set_text("train set embedding PHATE")
        plt.colorbar(im1, ax=ax[0])

        # VALID SET
        im2 = ax[1].scatter(valid_embed_phate[:,0],
                            valid_embed_phate[:,1],
                            c=valid_targs.numpy(),
                            s=5)

        ax[1].title.set_text("valid set embedding PHATE")
        plt.colorbar(im2, ax=ax[1])


        # TEST SET
        im3 = ax[2].scatter(test_embed_phate[:,0],
                            test_embed_phate[:,1],
                            c=test_targs.numpy(),
                            s=10)

        ax[2].title.set_text("test set embedding PHATE")
        plt.colorbar(im3, ax=ax[2])

        wandb_logger.experiment.log({"latent space PHATE  w fitness values": wandb.Image(plt)})


        # PHATE plot SEQ DISTANCES
        # -------------------------------------
        fig, ax = plt.subplots(1,3, figsize=(30,10))

        # TRAIN SET
        im1 = ax[0].scatter(train_embed_phate[:,0],
                            train_embed_phate[:,1],
                            c=data.train_split_seqd,
                            s=5)

        ax[0].title.set_text("train set embedding PHATE")
        plt.colorbar(im1, ax=ax[0])

        # VALID SET
        im2 = ax[1].scatter(valid_embed_phate[:,0],
                            valid_embed_phate[:,1],
                            c=data.valid_split_seqd,
                            s=5)

        ax[1].title.set_text("valid set embedding PHATE")
        plt.colorbar(im2, ax=ax[1])


        # TEST SET
        im3 = ax[2].scatter(test_embed_phate[:,0],
                            test_embed_phate[:,1],
                            c=data.test_split_seqd,
                            s=10)

        ax[2].title.set_text("test set embedding PHATE")
        plt.colorbar(im3, ax=ax[2])

        wandb_logger.experiment.log({"latent space PHATE w sequence distances": wandb.Image(plt)})
    except:
        print("PHATE run failed")


    print("getting smoothness values")

    eval_utils.get_all_smoothness_values(targets_list=targets_list,
                                        seqs_list = recon_targ_list,
                                        embeddings_list=embed_list,
                                        wandb_logger=wandb_logger)
    
    print("smoothness values logged")

    # ------------------------------------------------
    # FITNESS PREDICTION EVALUATIONS
    # ------------------------------------------------
 # check that model makes predictions
    print("running fitness prediction evaluations")

    eval_utils.get_all_fitness_pred_metrics(targets_list=targets_list,
                                        predictions_list=predictions_list,
                                        wandb_logger=wandb_logger)

    # ----------------------------
    # R2 PLOTS
    # ----------------------------
    print("creating R2 plots ")
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(train_fit_pred,
                train_targs, s=10, alpha=0.5)

    ax.set_xlabel('predicted enrichment', fontsize=10)
    ax.set_ylabel('true enrichment', fontsize=10)

    wandb_logger.experiment.log({" train set pred vs true enrichment": wandb.Image(plt)})

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(valid_fit_pred,
                valid_targs, s=10, alpha=0.5)

    ax.set_xlabel('predicted enrichment', fontsize=10)
    ax.set_ylabel('true enrichment', fontsize=10)

    wandb_logger.experiment.log({" valid set pred vs true enrichment": wandb.Image(plt)})

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(test_fit_pred,
                test_targs, s=10, alpha=0.5)

    ax.set_xlabel('predicted enrichment', fontsize=10)
    ax.set_ylabel('true enrichment', fontsize=10)

    wandb_logger.experiment.log({" test pred vs true enrichment": wandb.Image(plt)})


    # Reconstruction evaluations
    # ------------------------------

    print("running reconstruction evaluations")
    eval_utils.get_all_recon_pred_metrics(targets_list=recon_targ_list,
                                    predictions_list=recon_list,
                                    wandb_logger=wandb_logger)


    print("all evaluations complete")

    toc = time.perf_counter()
    print(f"training and evaluations finished in {toc - tic:0.4f} seconds")
