
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from phate import PHATE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


import torch 
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import relso.grad.models as hmodels
import relso.utils.eval_utils as heval_utils


def import_model_from_ckpt(model_name, path_to_ckpt_file):

    # get model
    proto_model = hmodels.str2model(model_name)

    loaded_model = proto_model.load_from_checkpoint(path_to_ckpt_file)

    print(f'loaded {model_name} model from {path_to_ckpt_file}')

    return loaded_model




def train_prediction_head(model, data, wandb_logger, e2e_train=False):
    """[summary]

    Args:
        model ([type]): [description]
        dataset ([type]): [description]
    """
    model.alpha_val = 1
    model.beta_val = 0
    model.gamma_val = 0
    model.eta_val = 0


    if e2e_train == False:
        # only unfreeze prediction head
        model.requires_grad_(False) 
        print('\n')
        for name, parameter in model.named_parameters():
            if 'regressor_module' in name:
                parameter.requires_grad = True

    else: 
        model.requires_grad_(True)
        model.train()
      
    early_stop_callback = EarlyStopping(
            monitor='valid_loss', # set in EvalResult
            min_delta=0.001,
            patience=8,
            verbose=True,
            mode='min'
            )


    trainer = pl.Trainer(max_epochs=50,
                        gpus=1,
                        callbacks=[early_stop_callback],
                        logger=wandb_logger,
                        gradient_clip_val=1,
                        auto_lr_find=True) 

    trainer.fit(model=model, 
                train_dataloader=data.train_dataloader(),
                val_dataloaders=data.valid_dataloader(),
                )

    model.eval()
 
    return model, trainer



def train_reconstruction_head(model, data, wandb_logger, e2e_train=False):
    """[summary]

    Args:
        model ([type]): [description]
        dataset ([type]): [description]
    """
    model.alpha_val = 0
    model.beta_val = 0
    model.gamma_val = 1
    model.eta_val = 0

    if e2e_train == False:
        # only unfreeze prediction head
        model.requires_grad_(False) 
        print('\n')
        for name, parameter in model.named_parameters():
            if 'dec_conv_module' in name:
                parameter.requires_grad = True

    else: 
        model.requires_grad_(True)
        model.train()
      
    early_stop_callback = EarlyStopping(
            monitor='valid_loss', # set in EvalResult
            min_delta=0.001,
            patience=8,
            verbose=True,
            mode='min'
            )


    trainer = pl.Trainer(max_epochs=50,
                        gpus=1,
                        callbacks=[early_stop_callback],
                        logger=wandb_logger,
                        gradient_clip_val=1,
                        auto_lr_find=True) 

    trainer.fit(model=model, 
                train_dataloader=data.train_dataloader(),
                val_dataloaders=data.valid_dataloader(),
                )

    model.eval()
 
    return model, trainer

    
# def plot_optim_traj(coords, fitness, n_steps, save_path):
#     """[summary]

#     Args:
#         coords ([type]): [description]
#         fitness ([type]): [description]
#         n_steps ([type]): [description]
#         save_path ([type]): [description]
#     """

#     fig, ax = plt.subplots(figsize=(10,10))

#     ax.scatter(coords[:,0][:-n_steps], coords[:,1][:-n_steps],
#              c=fitness[:-n_steps], s=3)

#     # start
#     ax.scatter(coords[:,0][-n_steps], coords[:,1][-n_steps],
#              s=40,c='k', marker='s')

#     # stop
#     ax.scatter(coords[:,0][-1], coords[:,1][-1], 
#                 s=40, c='k', marker='^')

#     # path
#     ax.plot(coords[:,0][-n_steps:], coords[:,1][-n_steps:], c='r')

#     plt.savefig(save_path)


# def plot_multiple_optim_traj_PCA(coords, fitness, optim_coords_list, traj_names, save_path):
#     """[summary]

#     Args:
#         coords ([type]): [description]
#         fitness ([type]): [description]
#         optim_coords_list ([type]): [description]
#         save_path ([type]): [description]
#     """

#     fig, ax = plt.subplots(figsize=(10,10), dpi=150)

#     ax.scatter(coords[:,0], coords[:,1],
#              c=fitness, s=3, alpha=0.4)

#     # start
#     traj_lines = []
#     colors = ['r', 'm','c']
#     for indx, trajcoords in enumerate(optim_coords_list):

#         ax.scatter(trajcoords[:,0][0], trajcoords[:,1][0],
#                 s=40,c='k', marker='s')

#         # stop
#         ax.scatter(trajcoords[:,0][-1], trajcoords[:,1][-1], 
#                     s=40, c='k', marker='^')

#         # path
#         traj, = ax.plot(trajcoords[:,0], trajcoords[:,1], c=colors[indx], label=traj_names[indx])
        
#         traj_lines.append(traj)



#     plt.legend(traj_lines, traj_names)
#     plt.savefig(save_path)


def plot_embedding(embeddings, fitness, wandb_logger, save_path, plot_type='PCA'):
    if plot_type == 'PCA':
        emb_coords = PCA(n_components=2).fit_transform(embeddings)
    
    elif plot_type == 'PHATE':
        emb_coords = PHATE(n_components=2).fit_transform(embeddings)

    elif plot_type == 'TSNE':
        emb_coords = TSNE(n_components=2).fit_transform(embeddings)


    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(emb_coords[:,0], emb_coords[:,1], c=fitness, s=3)
    plt.savefig(save_path)

    if wandb_logger:
        wandb_logger.experiment.log({f'Original Embedding - {plot_type}': wandb.Image(plt)})

    return emb_coords



def plot_multiple_optim_traj(embeddings, fitness, optim_embeddings_list, 
                            traj_names, save_path, run_indx, wandb_logger, plot_type='PCA'):
    """[summary]

    Args:
        embeddings ([type]): [description]
        fitness ([type]): [description]
        optim_embeddings_list ([type]): [description]
        save_path ([type]): [description]
    """
    
    num_trajs = len(optim_embeddings_list)

    all_embed_list = [embeddings] + optim_embeddings_list
    all_embeddings = np.concatenate(all_embed_list)

    embed_mask = []
    for indx, coords_set in enumerate(all_embed_list):
        embed_mask += [indx] * len(coords_set)

    embed_mask = np.array(embed_mask)

    # all_fitness_vals = np.concatenate([fitness] + )
    if plot_type == 'PCA':
        all_coords = PCA(n_components=2).fit_transform(all_embeddings)
    
    elif plot_type == 'PHATE':
        all_coords = PHATE(n_components=2).fit_transform(all_embeddings)
    
    elif plot_type == 'TSNE':
        all_coords = TSNE(n_components=2).fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    # background coords
    ax.scatter(all_coords[embed_mask == 0][:,0],
               all_coords[embed_mask == 0][:,1],
               c=fitness, s=3, alpha=0.6)

    # start
    traj_lines = []
    colors = ['r', 'm','c', 'y', 'magenta', 'brown', 'deeppink']
    markers = ['^', 'v', 'P', '*','D', 'X', 'o' ]
    for indx in range(1,num_trajs+1):

        coords_subset = all_coords[embed_mask == indx]

        ax.scatter(coords_subset[:,0][0],
                    coords_subset[:,1][0],
                   s=40,c='k', marker='s')

        # stop
        e_marks = ax.scatter(coords_subset[:,0][-1],
                   coords_subset[:,1][-1],
                    s=40, c='k', marker=markers[indx-1],
                    label=traj_names[indx-1])

        # path
        traj, = ax.plot(coords_subset[:,0],
                        coords_subset[:,1],
                        c=colors[indx-1],
                        label=traj_names[indx-1])
        
        traj_lines.append(traj)

    plt.legend(traj_lines, traj_names)
    plt.savefig(save_path)

    if wandb_logger:
        wandb_logger.experiment.log({f'LSO Optimization Trajectories {plot_type} seed={run_indx}': wandb.Image(plt)})

    plt.close()


def plot_mulitple_fitness_trajs(list_of_fit_trajs, list_of_traj_names, save_path, run_indx, wandb_logger=None):
    """plot fitness change for multiple series

    creates subplots, one for each fitness trajectory
    Args:
        list_of_fit_trajs ([type]): [description]
    """

    n_trajs = len(list_of_fit_trajs)
    # max_fit = max([max(x) for x in list_of_fit_trajs])
    fig, ax = plt.subplots(n_trajs, 1, figsize=(15, 3*n_trajs))

    traj_lines = []
    colors = ['r', 'm','c', 'y', 'magenta', 'brown', 'deeppink']
    
    for indx, traj in enumerate(list_of_fit_trajs):
       
        c_traj, = ax[indx].plot(np.arange(len(traj)), traj, 
            label=list_of_traj_names[indx],
            c=colors[indx])
 
        traj_lines.append(c_traj)


    fig.legend(handles=traj_lines,     # The line objects
           labels=list_of_traj_names,   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Fitness Optimization"  # Title for the legend
           )
    plt.subplots_adjust(right=0.85)

    plt.savefig(save_path)

    if wandb_logger:
        wandb_logger.experiment.log({f'LSO Fitness Trajectories seed={run_indx}': wandb.Image(plt)})

    plt.close()
    

def plot_landscape_3d(coords, fitness, save_path, plot_type='PCA', wandb_logger=None):
    """

    Args:
        embeddings ([type]): [description]
        fitness ([type]): [description]
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(coords[:,0],
                    coords[:,1],
                    fitness,
                    cmap='viridis', edgecolor='none')
    
    plt.savefig(save_path)

    if wandb_logger:
        wandb_logger.experiment.log({f'3D Embedding {plot_type}': wandb.Image(plt)})
    plt.close()

    
def plot_boxplot(fitness_array, x_labels, save_path, wandb_logger=None):
    """box plot util 

    pass in an list of max fitness values for algorithms over N runs
    
    fitness_array = M x N 

    M = number of algorithms
    N = number of runs
   

    """
    fig, ax = plt.subplots(figsize=(10,6))

    ax.boxplot(fitness_array)

    ax.set_xticklabels(x_labels)

    plt.savefig(save_path)

    if wandb_logger:
        wandb_logger.experiment.log({'LSO Box Plots': wandb.Image(plt)})

    plt.close()



# def plot_swarm_boxplot(fitness_array, x_labels, save_path):
#     """box plot util 

#     pass in an list of max fitness values for algorithms over N runs
    
#     fitness_array = M x N 

#     M = number of algorithms
#     N = number of runs
   

#     """
#     fig, ax = plt.subplots(figsize=(10,6))

#     ax.boxplot(fitness_array)

#     ax.set_xticklabels(x_labels)

#     plt.savefig(save_path)



def plot_embedding_end_points(embeddings, fitness, optim_embeddings, algo_name, save_path, plot_type = 'PCA', wandb_logger=None):
    """
    embed is of shape n_inits x n_steps x  x embed_dim


    Args:
        embeddings ([type]): [description]
        fitness ([type]): [description]
        optim_embeddings ([type]): [description]
        algo_name ([type]): [description]
        save_path ([type]): [description]
        plot_type (str, optional): [description]. Defaults to 'PCA'.
    """
    latent_dim = embeddings.shape[-1]
    num_trajs, len_traj, _ = optim_embeddings.shape
    print(f'num trajs: {num_trajs}')
    
    all_embeddings = np.concatenate([embeddings, optim_embeddings.reshape(-1, latent_dim)], 0)

    if plot_type == 'PCA':
        all_coords = PCA(n_components=2).fit_transform(all_embeddings)
    
    elif plot_type == 'PHATE':
        all_coords = PHATE(n_components=2).fit_transform(all_embeddings)

    elif plot_type == 'TSNE':
        all_coords = TSNE(n_components=2).fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(5,5), dpi=500)

    ax.title.set_text(algo_name)

    # background coords
    ax.scatter(all_coords[:-num_trajs * len_traj,0],
               all_coords[:-num_trajs * len_traj,1],
               c=fitness, s=4, alpha=0.6)

    for i in range(num_trajs):
        
        endpt_indx = -1 * ( (i * len_traj) + 1)


        # starting point
        ax.scatter(all_coords[endpt_indx,0],
                all_coords[endpt_indx,1],
                    s=40,c='k', marker='^')

    plt.savefig(save_path)
    
    if wandb_logger:
        wandb_logger.experiment.log({f'LSO ending points  {plot_type} {algo_name}': wandb.Image(plt)})



