"""
Optimization algorithms
"""



import numpy as np
import numpy.ma as ma
import numpy.linalg as LA
import copy

from tqdm import tqdm 

from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


import torch
import torch.nn as nn



"""

def grad_free_optimizer(initial_sequence, oracle, N):

    ...
    optimization step
    ...

    return opm

"""

######################
# Gradient Free Methods
######################


def eval_oracle(data_point, oracle):


    data_point = torch.Tensor(data_point).reshape(1,-1)
    # print("data point shape: {}".format(data_point.shape))

    with torch.no_grad():

        fit_value = oracle(data_point).squeeze().item()
    # print("fitness value: {}".format(fit_value))
    return fit_value


######################
# Directed Evolution
######################

def directed_evolution_sequence(initial_sequence, model, positions):
    """
    from https://www.pnas.org/content/pnas/116/18/8852.full.pdf
    
    traditional directed evolution by greedy walk
    1. saturation mutagenesis at single position
    2. fix optimal mutation
    3. repeat for all positions

    
    x = input_seq
    for pos i in [0,N-1]:
        best_aa optimial AA for pos i
        x[i] == best_aa
    return x

    initial_sequence = sequence of inds (1 x N)
    model = model where f(sequence) = y
    positions = iterable of positions
    """
    cand_seq = torch.from_numpy(initial_sequence).unsqueeze(0)

    with torch.no_grad():
        initial_fit = model(cand_seq)[0][-1]
    
    cand_traj = np.zeros((len(positions)+1, len(initial_sequence)))
    
    fit_traj = np.zeros((len(positions) + 1))
    
    cand_traj[0] = cand_seq.reshape(-1).numpy()
    fit_traj[0] = initial_fit

    for indx, pos in enumerate(tqdm(positions)):
        
        # create full expansion at position
        cand_seqs = cand_seq.repeat(22, 1)    
        cand_seqs[:, pos] = torch.arange(22)
        
        # screen expansion
        with torch.no_grad():
            cand_seqs_fit = model(cand_seqs)[0][-1]

        max_fit_aa_indx = cand_seqs_fit.argmax(0)
              
        max_fit = cand_seqs_fit.max()
        cand_seq = cand_seqs[max_fit_aa_indx]
        
        cand_traj[indx+1] = cand_seq.reshape(-1).numpy()
        fit_traj[indx+1] = max_fit.numpy()
        
    return cand_traj, fit_traj


############################
# MCMC
############################


# Sequence Level
# -------------------------
def model_predict(sequence, model):
    
    model = model.eval()
    
    if type(sequence) == type(np.ones(1)):
        sequence = torch.from_numpy(sequence)
        
    sequence = sequence.reshape(1,-1)
    
    with torch.no_grad():
        fit = model(sequence)[0][-1].numpy().squeeze()
        
    return fit
    
def mutate_sequence(sequence, num_mutations):
    
    sequence = sequence.copy()
        
    AA_inds = np.arange(22)

    positions = np.random.choice(np.arange(len(sequence)), num_mutations)

    for pos in positions:
        pos_val = sequence[pos]
        
        if pos_val == 21:
            print('mutation in padding region - ignoring')
        else:
            # change current AA to a new AA
            mut_choices = np.ma.masked_where(AA_inds == pos_val, AA_inds)
            chosen_mut = np.random.choice(AA_inds[~mut_choices.mask],1)


            sequence[pos] = chosen_mut

    return sequence



def acceptance_step(curr_fit, prop_fit, T):
    """
    returns bool
    
    """
    out_dict = {0: False, 1: True}

    # acceptance probability
    prob = np.exp(( (prop_fit-curr_fit) / T))

    if prob == np.nan:
        outcome = 0

    else:
        prob = min(1,prob)
        outcome = np.random.binomial(1, prob)

    return out_dict[outcome]       


def get_l1_norm(seq1, seq2):
    
    # convert to one-hot
    seq1, seq2 = seq1.squeeze(), seq2.squeeze()
    seq1 = np.eye(22)[seq1]
    seq2 = np.eye(22)[seq2]    
    l1_norm = LA.norm(seq1 - seq2, 1)
    
    return l1_norm
    

def metropolisMCMC_sequence(initial_sequence, model, T=0.01, mu=1, trust_radius=15,
                            N_steps=20):
    """
    from pg 24 of low-N
    https://www.biorxiv.org/content/10.1101/2020.01.23.917682v2.full.pdf

    """
    # start at initial sequence

    curr_seq = initial_sequence.numpy()
    curr_fit = model_predict(curr_seq, model)
    
    cand_traj = np.zeros( (N_steps,  len(initial_sequence) ))
    fit_traj = np.zeros((N_steps))
    
    cand_traj[0] = curr_seq.reshape(-1)
    fit_traj[0] = curr_fit
    
    
    # optimization loop
    for step_indx in tqdm(range(1,N_steps)): 

        num_mut = np.random.poisson(mu)
        
        # produce candidate
        prop_seq = mutate_sequence(curr_seq, num_mut)
        
        # selection step

        if get_l1_norm(prop_seq, curr_seq) < trust_radius: # mut radius
            
            prop_fit = model_predict(prop_seq, model)

            if acceptance_step(curr_fit, prop_fit, T):
#                 print('change accepted')
                curr_seq = prop_seq.copy()
                curr_fit = prop_fit.copy()
            
        # logging
        cand_traj[step_indx] = curr_seq.reshape(-1)
        fit_traj[step_indx] = curr_fit

    return cand_traj, fit_traj



# Latent space 
# -------------------------


def metropolisMCMC_embedding(initial_embedding, oracle,
                             T=0.01, delta=0.1, N_steps=1000):
    """
    MCMC on a continous vector space 
    proposed candidates come from random direction
    

    """
    embed_dim = initial_embedding.shape[-1]

    # start at initial sequence
    curr_embedding = initial_embedding.reshape(1,embed_dim)
    curr_fit = eval_oracle(curr_embedding, oracle)

    print("starting fitness: {}".format(curr_fit))
    fitness_list = [curr_fit]

    
    out_embedding_array = np.zeros((N_steps, embed_dim))
    out_embedding_array[0] = curr_embedding

    # optimization loop
    for indx in tqdm(range(1,N_steps)): 

        prop_embedding = curr_embedding + delta * np.random.randn(embed_dim)

        prop_fit = eval_oracle(prop_embedding, oracle)

        if acceptance_step(curr_fit,prop_fit, T):
            curr_embedding = prop_embedding
            curr_fit = prop_fit
        
        # logging
        fitness_list.append(curr_fit)
        out_embedding_array[indx] = curr_embedding


    return out_embedding_array, np.array(fitness_list)



def model_cycle(embedding, model):
    """passes embedding through decoder and encoder

    Args:
        embedding ([type]): [description]
        model ([type]): [description]
    """
    with torch.no_grad():
        embedding = torch.from_numpy(embedding).float().reshape(1,-1)
        
        decoded_seq = model.decode(embedding).argmax(1)

        re_embed = model.encode(decoded_seq).numpy()

    return re_embed



def metropolisMCMC_embedding_cycle(initial_embedding, oracle, model,
                             T=0.01, delta=0.05, N_steps=1000, perturbation=True):
    """
    MCMC on a continous vector space 
    proposed candidates come from random direction
    """
    embed_dim = initial_embedding.shape[-1]

    # start at initial sequence
    curr_embedding = initial_embedding.reshape(1,embed_dim)
    curr_fit = eval_oracle(curr_embedding, oracle)

    print("starting fitness: {}".format(curr_fit))
    fitness_list = [curr_fit]

    
    out_embedding_array = np.zeros((N_steps, embed_dim))
    out_embedding_array[0] = curr_embedding

    # optimization loop
    for indx in tqdm(range(1,N_steps)): 

        # perturbation step
        if perturbation:
            prop_embedding = curr_embedding + delta * np.random.randn(embed_dim)
            prop_embedding = model_cycle(prop_embedding, model)

        else:
            prop_embedding = model_cycle(curr_embedding, model)

        prop_fit = eval_oracle(prop_embedding, oracle)

        if acceptance_step(curr_fit,prop_fit, T):
            curr_embedding = prop_embedding
            curr_fit = prop_fit
        
        # logging
        fitness_list.append(curr_fit)
        out_embedding_array[indx] = curr_embedding


    return out_embedding_array, np.array(fitness_list)


############################
# Hill Climbing
############################


def get_knn_directions(dataset, current_point, k):
    dists_to_initial_point = distance.cdist(current_point.reshape(1,-1), dataset).flatten()

    knn = [dataset[x] for x in np.argsort(dists_to_initial_point)[:k]]
    return knn        


def get_steepest_neighbor(neighbors, oracle):

    fitness_values = np.array([eval_oracle(x, oracle) for x in neighbors])
    steepest_neighbor = neighbors[fitness_values.argmax()]

    return steepest_neighbor, max(fitness_values)

def get_stochastic_steepest_neighbor(neighbors, oracle, curr_fit):

    fitness_values = np.array([eval_oracle(x, oracle) for x in neighbors])

    incline_inds = np.arange(len(fitness_values))[fitness_values > curr_fit]

    if len(incline_inds) == 0:
        all_inds = np.arange(len(fitness_values))
        choice_ind = np.random.choice(all_inds, 1)[0]

    else:    
        choice_ind = np.random.choice(incline_inds, 1)[0]

    choice_neighbor = neighbors[choice_ind]
    choice_fit = fitness_values[choice_ind]

    return choice_neighbor, choice_fit



def nn_hill_climbing_embedding(initial_embedding, oracle, dataset_embeddings,
                            step_interp=0.5, k_neighbors=30, N_steps=1000, stochastic=False):
    """[summary]

    Args:
        initial_embedding ([type]): [description]
        oracle ([type]): [description]
        step_interp ([type]): [description]
        N_steps ([type]): [description]
    """

    embed_dim = initial_embedding.shape[-1]

    curr_embedding = initial_embedding.reshape(1,embed_dim)
    curr_fit = eval_oracle(curr_embedding, oracle)
    
    print("starting fitness: {}".format(curr_fit))

    fitness_list = [curr_fit]

    out_embedding_array = np.zeros((N_steps, embed_dim))
    out_embedding_array[0] = curr_embedding

    for indx in tqdm(range(1,N_steps)):

        # search step
        k_directions = get_knn_directions(dataset_embeddings, curr_embedding, k_neighbors )

        if stochastic:
            next_neighbor, next_fitness = get_stochastic_steepest_neighbor(k_directions, oracle, curr_fit)
        else:
            next_neighbor, next_fitness = get_steepest_neighbor(k_directions, oracle)

        next_direction = next_neighbor - curr_embedding

        # update step
        curr_embedding += step_interp * next_direction
        curr_fit = next_fitness

        # logging 
        out_embedding_array[indx] = curr_embedding
        fitness_list.append(curr_fit)

    return out_embedding_array, np.array(fitness_list)

############################
# Gradient Methods
############################

def grad_ascent(initial_embedding, model, N_steps, lr, cycle=False,
                 noise=False, sigma=0.001):
    
    #need to pass the sequence through the network layers for gradient to be taken, so cycle the embedding once
    model.requires_grad_(True)
    grad_list = []
    
    
    # data logging
    embed_dim = initial_embedding.shape[-1]
    out_embedding_array = np.zeros((N_steps, embed_dim))
    out_fit_array = np.zeros((N_steps))
    
    # initial step
    curr_embedding = torch.tensor(initial_embedding, requires_grad=True).reshape(-1, embed_dim)
    curr_fit = model.regressor_module(curr_embedding)
    
    #print("starting fitness: {}".format(curr_fit))

    # save step 0 info
    out_embedding_array[0] = curr_embedding.reshape(1,embed_dim).detach().numpy()
    out_fit_array[0] = curr_fit.detach().numpy()
    
    assert curr_embedding.requires_grad

    for step in tqdm(range(1,N_steps)):
        model.train()
        
        grad = torch.autograd.grad(curr_fit, curr_embedding)[0] # get gradient

        if noise:
            grad += torch.normal(torch.zeros(grad.size()),sigma)     

        grad_list.append(grad.detach())

        # curr_embedding = curr_embedding.detach()
        # update step
        update_step = grad * lr
        curr_embedding = curr_embedding +  update_step 
        
        # cycle bool
        model = model.eval()
        
        if cycle:
            nseq = model.decode(curr_embedding).argmax(1)
            curr_embedding = model.encode(nseq)

        curr_fit = model.regressor_module(curr_embedding)
        
        # save step i info
        out_embedding_array[step] = curr_embedding.detach().numpy()
        out_fit_array[step] = curr_fit.detach().numpy()
        
    return out_embedding_array, out_fit_array
