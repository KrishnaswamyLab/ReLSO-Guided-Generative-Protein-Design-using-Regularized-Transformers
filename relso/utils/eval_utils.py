"""
Helper functions for evaluation
"""
import numpy as np

import torch
from torch import nn

import scipy
from scipy import stats
from scipy.spatial import distance as scidist
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import hamming

import networkx as nx

from numpy import linalg as LA

from tqdm import tqdm




""" Calculate R^2 value for seaborn scatterplots """
def get_pearson_r2(x, y):
    return stats.pearsonr(x, y)[0]

def get_spearman_r2(x, y):
    return stats.spearmanr(x, y)[0]


"""
Get graph dirchlet energy
"""

def get_smoothnes_rbf(embeddings, energies, beta):
    """use rbf-based affinity matrix for smoothness calc

    Args:
        embeddings ([type]): [description]
        energies ([type]): [description]
        beta ([type]): [description]

    Returns:
        [type]: [description]
    """

    N = embeddings.shape[0]

    energies = energies.reshape(N,1)

    # compute distances
    print("computing distances")
    dist_mat = scidist.squareform(scidist.pdist(embeddings))

    # get affinity matrix
    aff_mat = np.exp(-beta * dist_mat / dist_mat.std())

    print("aff mat: {}".format(aff_mat.shape))

    nn_graph = nx.from_numpy_array(aff_mat)

    print("computing combinatorial graph laplacian")
    L_mat = nx.laplacian_matrix(nn_graph).todense()

    # compute smoothness index
    print("computing smoothness value")
    lap_smooth = np.matmul(L_mat, energies)
    lap_smooth = np.matmul(energies.T, lap_smooth)
    signal_dot = np.matmul(energies.T, energies)
    lap_smooth = lap_smooth / signal_dot

    print("smoothness for beta={}: {}".format(beta, lap_smooth.item()))

    return lap_smooth.item()


def get_smoothnes_kNN_sparse(embeddings, energies, K=5):
    """ kNN based graph for smoothness calc
    Args:
        embeddings ([type]): [description]
        energies ([type]): [description]
        K ([type]): [description]
    Returns:
        [type]: [description]
    """

    N = embeddings.shape[0]

    energies = energies.reshape(N,1)

    # get kNN graph
    print("getting kNN graph")
    A = kneighbors_graph(embeddings, n_neighbors=K,
                                  mode='connectivity')
    A_coo = A.tocoo()

    # compute smoothness index
    print("computing smoothness value")
    lap_smooth = 0

    for i, j in zip(A_coo.row, A_coo.col):
        if i < j and A[j, i] == 1:
            lap_smooth += (energies[i] - energies[j])**2

    print("smoothness for K={}: {}".format(K, lap_smooth.item()))

    return lap_smooth.item()


def get_smoothnes_knn_weighted(embeddings, energies, K=5):
    """use knn graph with weighted edges

    Args:
        embeddings ([type]): [description]
        energies ([type]): [description]
        beta ([type]): [description]

    Returns:
        [type]: [description]
    """

    N = embeddings.shape[0]

    energies = energies.reshape(N,1)

    # get kNN graph
    print("getting kNN graph with distances")
    A = kneighbors_graph(embeddings, n_neighbors=K,
                                  mode='distance')
    A_coo = A.tocoo()

    # get gamma val
    print('calculating gamma')
    gamma_val = get_gamma_from_sparse(A)

    # compute smoothness index
    print("computing smoothness value")
    binary_lap_smooth = 0
    weighted_lap_smooth = 0

    for i, j in zip(A_coo.row, A_coo.col):
        if i < j and A[i,j] != 0:

            sqrd_diff = (energies[i] - energies[j])**2 # squared diff

            # binary case
            binary_lap_smooth += sqrd_diff

            # weighted case
            edge_weight = np.exp(-gamma_val * A[i,j]) # get edge weight
            weighted_lap_smooth += edge_weight * sqrd_diff

    print(" binary smoothness for K={}: {}".format(K, binary_lap_smooth.item()))
    print(" weighted smoothness for K={}: {}".format(K, weighted_lap_smooth.item()))

    return binary_lap_smooth.item(), weighted_lap_smooth.item()


def get_binary_knn_smooth(embeddings, fitness_vec, K=5):
     """use knn graph with weighted edges

     Args:
         embeddings ([type]): [description]
         energies ([type]): [description]
         beta ([type]): [description]

     Returns:
         [type]: [description]
     """

     N = embeddings.shape[0]

     fitness_vec = fitness_vec.reshape(N,1)

     # get kNN graph
     A = kneighbors_graph(embeddings, n_neighbors=K,
                                   mode='distance')
     A_coo = A.tocoo()

     b_smooth_fit = 0

     for i, j in zip(A_coo.row, A_coo.col):
         if i < j and A[i,j] != 0:

             fit_sqrd_diff = (fitness_vec[i] - fitness_vec[j])**2 # squared dif

             # binary case
             b_smooth_fit += fit_sqrd_diff


     return b_smooth_fit

def get_gamma_from_sparse(sparse_matrix):
    """calc gamma param from sparse matrix


    from https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    gamma is equal to (1/(2*(std)**2))

    Args:
        sparse_matrix ([type]): [description]
    """
    n_rows = sparse_matrix.shape[0]

    sum_var = 0
    avg_distance = 0
    for i in range(n_rows):

        # use only non-zero entries
        # to avoid artificial depression of std
        dense_row = sparse_matrix.getrow(i).todense()
        nonz_inds = dense_row.nonzero()[1]
        sum_var += dense_row[:, nonz_inds].std()**2

        avg_distance += dense_row[:, nonz_inds].mean()


    avg_distance = avg_distance / n_rows
    print("avg distance: {}".format(avg_distance))

    # sum_std = std_dev / n_rows

    corr_var = sum_var / (n_rows - 1 )
    print("sum var: {}".format(sum_var))
    print("corr var: {}".format(corr_var))

    gamma = 2 * corr_var
    gamma = gamma ** (-1)

    print("gamma value determined as: {}".format(gamma))

    return gamma


def get_avg_distance(embeddings, k):

    n_obs = embeddings.shape[0]

    sub_inds = np.random.choice(np.arange(n_obs), int(0.1 * n_obs))

    sub_embeddings = embeddings[sub_inds]

    A = kneighbors_graph(sub_embeddings, n_neighbors=k,
                                mode='distance')
    sparse_matrix = A.tocoo()

    n_rows = sparse_matrix.shape[0]

    sum_std = 0
    avg_distance = 0
    for i in tqdm(range(n_rows)):

        # use only non-zero entries
        # to avoid artificial depression of std
        dense_row = sparse_matrix.getrow(i).todense()
        nonz_inds = dense_row.nonzero()[1]
        sum_std += dense_row[:, nonz_inds].std()

        avg_distance += dense_row[:, nonz_inds].mean()


    avg_distance = avg_distance / n_rows

    return avg_distance

def get_percent_error(pred_values, targ_values):

    perc_error_array = np.zeros(shape=(pred_values.flatten().shape[0], 1))

    for indx, (pred, targ) in enumerate(tqdm(zip(pred_values, targ_values))):
        perc_error = (pred - targ) / (targ + 0.0001)
        perc_error = 100 * np.abs(perc_error)

        perc_error_array[indx] = perc_error

    return perc_error_array.mean()






def get_all_smoothness_values(targets_list, seqs_list,  embeddings_list, wandb_logger):

    # train_targs, valid_targs, test_targs = targets_list
    # train_embed, valid_embed, test_embed = embeddings_list
    # train_seqd, valid_seqd, test_seqd = seqd_list


    print("getting smoothness values")

    print("calculating KNN graphs")
    A_mat_list = [kneighbors_graph(x, n_neighbors=5, metric='euclidean', mode='connectivity') for x in embeddings_list]


    print("fitness smoothness values")
    fit_smooth_vals = [get_fit_smoothness(A_mat_list[i], targets_list[i]) for i in range(3)]

    wandb_logger.log_metrics({'train_smooth_fb_k5': fit_smooth_vals[0]})
    wandb_logger.log_metrics({'valid_smooth_fb_k5': fit_smooth_vals[1]})
    wandb_logger.log_metrics({'test_smooth_fb_k5': fit_smooth_vals[2]})


    print("seq smoothness values")
    seq_smooth_vals = [get_seq_smoothness(A_mat_list[i], seqs_list[i]) for i in range(3)]

    wandb_logger.log_metrics({'train_smooth_sb_k5': seq_smooth_vals[0]})
    wandb_logger.log_metrics({'valid_smooth_sb_k5': seq_smooth_vals[1]})
    wandb_logger.log_metrics({'test_smooth_sb_k5': seq_smooth_vals[2]})






def get_all_fitness_pred_metrics(targets_list, predictions_list, wandb_logger):

    train_targs, valid_targs, test_targs = targets_list
    train_fit_pred, valid_fit_pred, test_fit_pred = predictions_list

    # R2
    print("calculating pearson r ")
    train_p_r_val = get_pearson_r2(train_targs.numpy().flatten(),
                        train_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"train pearson r":train_p_r_val})

    valid_p_r_val = get_pearson_r2(valid_targs.numpy().flatten(),
                        valid_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"valid pearson r":valid_p_r_val})

    test_p_r_val = get_pearson_r2(test_targs.numpy().flatten(),
                        test_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"test pearson r":test_p_r_val})



    print("calculating spearman r ")
    train_s_r_val = get_spearman_r2(train_targs.numpy().flatten(),
                        train_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"train spearman r":train_s_r_val})

    valid_s_r_val = get_spearman_r2(valid_targs.numpy().flatten(),
                        valid_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"valid spearman r":valid_s_r_val})

    test_s_r_val = get_spearman_r2(test_targs.numpy().flatten(),
                        test_fit_pred.numpy().flatten())
    wandb_logger.experiment.log({"test spearman r ":test_s_r_val})


    # MSE
    print("calculating R2 ")
    train_mse = nn.MSELoss()(train_fit_pred.squeeze(), train_targs.squeeze())
    wandb_logger.log_metrics({'train_enrichment_mse':train_mse.numpy()})

    valid_mse = nn.MSELoss()(valid_fit_pred.squeeze(), valid_targs.squeeze())
    wandb_logger.log_metrics({'valid_enrichment_mse':valid_mse.numpy()})

    test_mse = nn.MSELoss()(test_fit_pred.squeeze(), test_targs.squeeze())
    wandb_logger.log_metrics({'test_enrichment_mse':test_mse.numpy()})

    # L1
    print("calculating L1 ")
    train_l1 = nn.L1Loss()(train_fit_pred.squeeze(), train_targs.squeeze())
    wandb_logger.log_metrics({'train_enrichment_l1':train_l1.numpy()})

    valid_l1 = nn.L1Loss()(valid_fit_pred.squeeze(), valid_targs.squeeze())
    wandb_logger.log_metrics({'valid_enrichment_l1':valid_l1.numpy()})

    test_l1 = nn.L1Loss()(test_fit_pred.squeeze(), test_targs.squeeze())
    wandb_logger.log_metrics({'test_enrichment_l1':test_l1.numpy()})



def get_all_recon_pred_metrics(targets_list, predictions_list, wandb_logger):

    train_targs, valid_targs, test_targs = targets_list
    train_fit_pred, valid_fit_pred, test_fit_pred = predictions_list

    # R2
    print("calculating CE ")
    train_CE_val = nn.CrossEntropyLoss()(predictions_list[0], targets_list[0])

    wandb_logger.experiment.log({"train CE":train_CE_val})

    valid_CE_val = nn.CrossEntropyLoss()(predictions_list[1], targets_list[1])

    wandb_logger.experiment.log({"valid CE":valid_CE_val})

    test_CE_val = nn.CrossEntropyLoss()(predictions_list[2], targets_list[2])

    wandb_logger.experiment.log({"test CE":test_CE_val})



    print("calculating perplexity ")

    wandb_logger.experiment.log({"train perplexity": torch.exp(train_CE_val)} )
    wandb_logger.experiment.log({"valid perplexity": torch.exp(train_CE_val)} )
    wandb_logger.experiment.log({"test perplexity": torch.exp(train_CE_val)} )


def get_model_outputs(model, sequences):
    model.eval()
    
    data_loader = torch.utils.data.DataLoader(sequences, batch_size=200, shuffle=False)

    recon_preds = []
    fit_preds = []
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            preds, zrep = model(batch) # preds = [x_hat, y_hat]
            x_hat, y_hat = preds

            recon_preds.append(x_hat)
            fit_preds.append(y_hat)
            embeddings.append(zrep)

    recon_preds = torch.cat(recon_preds)
    fit_preds = torch.cat(fit_preds)
    embeddings = torch.cat(embeddings)

    return (recon_preds, fit_preds), embeddings



### Smoothness Utils
def get_fit_smoothness(A_mat, fitness):
    N = A_mat.shape[0]
    
    A_coo = A_mat.tocoo()
    
    
    binary_lap_smooth = 0
    for i, j in zip(A_coo.row, A_coo.col):

        sqrd_diff = (fitness[i] - fitness[j])**2 # squared diff

        # binary case
        binary_lap_smooth += sqrd_diff


    binary_lap_smooth = binary_lap_smooth/ N

    return binary_lap_smooth.item() 



def get_seq_smoothness(A_mat, sequences):
    N = A_mat.shape[0]
    
    A_coo = A_mat.tocoo()
    
    
    binary_lap_smooth = 0
    
    for i, j in zip(A_coo.row, A_coo.col):

        seq_dist = hamming(sequences[i],sequences[j])

        # binary case
        binary_lap_smooth += seq_dist


    binary_lap_smooth = binary_lap_smooth/ N

    return binary_lap_smooth.item() 