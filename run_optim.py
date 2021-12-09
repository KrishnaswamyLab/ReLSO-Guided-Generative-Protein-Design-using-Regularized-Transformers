
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


if __name__ == '__main__':

    parser = ArgumentParser(add_help=True)

    # required arguments
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--embeddings', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--weights', required=True, type=str)
    parser.add_argument('--n_steps', default=200, type=int)
    parser.add_argument('--log_dir', default='optim_logs/', type=str)
    parser.add_argument('--log_iter', default=None, type=int)
    parser.add_argument('--project_name', default='relso-optim', type=str)
    parser.add_argument('--det_inits', default=False, action='store_true')
    parser.add_argument('--model_evals', default=False, action='store_true')
    parser.add_argument('--alpha', required=False, type=float)
    parser.add_argument('--delta', required=False, default='adaptive', type=str)
    parser.add_argument('--k', required=False, default=5, type=float)

    cl_args = parser.parse_args()

    # logging
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")

    if cl_args.log_dir == 'e2e':
        save_dir =  f'end2end_logs/{cl_args.model}_{cl_args.dataset}_{cl_args.alpha}/optim_results/'

    elif cl_args.log_dir:
        save_dir = cl_args.log_dir + f'{cl_args.model}/{cl_args.dataset}/ns{cl_args.n_steps}/{date_suffix}/'

    else:
        save_dir = f'optim_logs/{cl_args.model}/{cl_args.dataset}/ns{cl_args.n_steps}/{date_suffix}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    # load embeddings
    embeddings = np.load(cl_args.embeddings)
    print(f'embeddings loaded with shape: {embeddings.shape}')


    # if alpha_val = 0, train prediction head
    if model.alpha_val == 0:
        model, _ = utils.train_prediction_head(model, data, wandb_logger)


    # get model evaluations
    # get prediction evaluations
    if cl_args.model_evals:
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


    # randomly initialize point
    n_steps = cl_args.n_steps
    num_inits = 30
    num_optim_algs = 7
    optim_algo_names = ['MCMC', 'MCMC-cycle', 'MCMC-cycle-noP','Hill Climbing', 'Stochastic Hill Climbing','Gradient Ascent','Gradient Ascent Cycle']

    optim_embedding_traj_array = np.zeros((num_inits, num_optim_algs, n_steps,  embeddings.shape[-1]))
    optim_fitness_traj_array = np.zeros((num_inits, num_optim_algs, n_steps))

    if cl_args.det_inits:
        print('deterministic seeds selected!')
        seed_vals = np.linspace(0,len(embeddings)-1, num_inits)
    else:
        print('random seeds selected!')
        seed_vals = np.random.choice(np.arange(len(embeddings)), num_inits)


    if cl_args.delta == 'adaptive':
        print('adaptive delta selected - computing delta based off pairwise distances')
        cl_args.delta = eval_utils.get_avg_distance(embeddings=embeddings, k=cl_args.k)

    else:
        cl_args.delta = float(cl_args.delta)

    for run_indx, init_indx in enumerate(seed_vals):

        init_indx = int(init_indx)

        print(f'\nrunning initialization {run_indx}/{num_inits}\n')

        init_point = embeddings[init_indx].copy()

        # MCMC
        print("\n")
        embedding_array_mcmc, fitness_array_mcmc = optim_algs.metropolisMCMC_embedding(initial_embedding=init_point.copy(),
                                                                                        oracle=model.regressor_module,
                                                                                       delta=cl_args.delta,
                                                                                       N_steps=n_steps)
        print(f'shape of output embedding array: {embedding_array_mcmc.shape}')
        print('init embed from output: {}'.format(embedding_array_mcmc[0][:10]))

        print("\n")
        embedding_array_mcmc_cycle, fitness_array_mcmc_cycle = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                            model=model,
                                                                                            delta=cl_args.delta,
                                                                                            N_steps=n_steps)
        print(f'shape of output embedding array: {embedding_array_mcmc_cycle.shape}')
        print('init embed from output: {}'.format(embedding_array_mcmc_cycle[0][:10]))


        print("\n")
        embedding_array_mcmc_cycle_noP, fitness_array_mcmc_cycle_noP = optim_algs.metropolisMCMC_embedding_cycle(initial_embedding=init_point.copy(),
                                                                                            oracle=model.regressor_module,
                                                                                            model=model,
                                                                                            N_steps=n_steps,
                                                                                            delta=cl_args.delta,
                                                                                            perturbation=False)
        print(f'shape of output embedding array: {embedding_array_mcmc_cycle_noP.shape}')
        print('init embed from output: {}'.format(embedding_array_mcmc_cycle_noP[0][:10]))


        # Hill climbing
        print("\n")
        embedding_array_hill, fitness_array_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding=init_point.copy(),
                                                                                        oracle=model.regressor_module,
                                                                                        dataset_embeddings=embeddings,
                                                                                        N_steps=n_steps)
        print(f'shape of output embedding array: {embedding_array_hill.shape}')
        print('init embed from output: {}'.format(embedding_array_hill[0][:10]))

        # Stochastic climbing
        print("\n")
        embedding_array_s_hill, fitness_array_s_hill = optim_algs.nn_hill_climbing_embedding(initial_embedding = init_point.copy(),
                                                                                            oracle = model.regressor_module,
                                                                                            dataset_embeddings=embeddings,
                                                                                            N_steps=n_steps,
                                                                                            stochastic=True)
        print(f'shape of output embedding array: {embedding_array_s_hill.shape}')
        print('init embed from output: {}'.format(embedding_array_s_hill[0][:10]))

# Gradient Ascent
        print("\n")
        embedding_array_ga, fitness_array_ga = optim_algs.grad_asct(initial_embedding = init_point.copy(),
                                                                    oracle=model.regressor_module,
                                                                    model=model,
                                                                    fullemb=embeddings,
                                                                    k=n_steps,
                                                                    lr=0.1,
                                                                    cycle=False)

        print(f'shape of output embedding array: {embedding_array_ga.shape}')
        print('init embed from output: {}'.format(embedding_array_ga[0][:10]))

# Gradient ascent cycle
        print("\n")
        embedding_array_gac, fitness_array_gac = optim_algs.grad_asct(initial_embedding = init_point.copy(),
                                                                    oracle=model.regressor_module,
                                                                    model=model,
                                                                    fullemb=embeddings,
                                                                    k=n_steps,
                                                                    lr=0.1,
                                                                    cycle=True)

        print(f'shape of output embedding array: {embedding_array_gac.shape}')
        print('init embed from output: {}'.format(embedding_array_gac[0][:10]))


        run_optim_embeddings = [embedding_array_mcmc,
                                embedding_array_mcmc_cycle,
                                embedding_array_mcmc_cycle_noP,
                                embedding_array_hill,
                                embedding_array_s_hill,
                                embedding_array_ga,
                                embedding_array_gac
                                ]
        run_optim_fitnesses = [fitness_array_mcmc,
                                    fitness_array_mcmc_cycle,
                                    fitness_array_mcmc_cycle_noP,
                                    fitness_array_hill,
                                    fitness_array_s_hill,
                                    fitness_array_ga,
                                    fitness_array_gac
                               ]

        for alg_indx, (embed, fit) in enumerate(zip(run_optim_embeddings, run_optim_fitnesses )):
            optim_embedding_traj_array[run_indx, alg_indx] = embed
            optim_fitness_traj_array[run_indx, alg_indx] = fit



        if cl_args.log_iter and run_indx % cl_args.log_iter == 0:

            # save trajs as plots
            utils.plot_mulitple_fitness_trajs(run_optim_fitnesses,
                                        optim_algo_names,
                                        run_indx=init_indx,
                                        wandb_logger=wandb_logger,
                                        save_path = save_dir +  f'optim_fitness_traj_all_run_{init_indx}.png')

            # PCA
            utils.plot_multiple_optim_traj(embeddings=embeddings,
                                fitness=train_targs,
                                optim_embeddings_list=run_optim_embeddings,
                                traj_names=optim_algo_names,
                                plot_type='PCA',
                                run_indx=init_indx,
                                wandb_logger=wandb_logger,
                                save_path=save_dir + f'optim_embedding_path_all_pca_run_{init_indx}.png')

            try:
                utils.plot_multiple_optim_traj(embeddings=embeddings,
                                    fitness=train_targs,
                                    optim_embeddings_list=run_optim_embeddings,
                                    traj_names=optim_algo_names,
                                    plot_type='PHATE',
                                    run_indx=init_indx,
                                    wandb_logger=wandb_logger,
                                    save_path=save_dir + f'optim_embedding_path_all_phate_run_{init_indx}.png')
            except:
                print("phate run failed")

            try:
                utils.plot_multiple_optim_traj(embeddings=embeddings,
                                    fitness=train_targs,
                                    optim_embeddings_list=run_optim_embeddings,
                                    traj_names=optim_algo_names,
                                    plot_type='TSNE',
                                    run_indx=init_indx,
                                    wandb_logger=wandb_logger,
                                    save_path=save_dir + f'optim_embedding_path_all_tsne_run_{init_indx}.png')
            except:
                print("phate run failed")


    # save embeddings
    print("saving embeddings")
    np.save(save_dir + 'optimization_embeddings.npy', optim_embedding_traj_array)
    np.save(save_dir + 'optimization_fitnesses.npy', optim_fitness_traj_array)



    # max fitnesss array shape: num_algos x num_runs
    print("logging max fitness values")
    max_fitness_array = optim_fitness_traj_array[:, :,  -1] # n_init x n_algo array
    utils.plot_boxplot(max_fitness_array, optim_algo_names,
                wandb_logger=wandb_logger,
                save_path= save_dir + f'max_fitness_boxplot.png')

    # log max fitness values
    # optim_fitness_traj_array shape: n_inits x n_algos x n_steps
    per_algo_fitness_values = optim_fitness_traj_array.transpose(1,0,2).reshape(len(optim_algo_names), -1)
    print(f'len of optim_algo_names: {len(optim_algo_names)}')
    print(f'len of max_fitness_array: {len(per_algo_fitness_values)}')

    for name, fit_vals in zip(optim_algo_names, per_algo_fitness_values):
        max_fit_i = fit_vals.max()
        wandb_logger.experiment.log({f'Max Fitness for {name} Runs': max_fit_i})


    endpoint_embed_array = optim_embedding_traj_array.transpose(1,0,2,3)
    # shape will now be num_algo x n_steps x n_inits x embed_dim


    for indx, (name, embeds) in enumerate(zip(optim_algo_names, endpoint_embed_array)):

        print(f'shape of embeddings: {embeddings.shape}')
        print(f'shape of embeds: {embeds.shape}')

        utils.plot_embedding_end_points(embeddings, train_targs, embeds, algo_name=name,
                    wandb_logger=wandb_logger,
                    save_path=save_dir + f'max_fitness_PCA_end_points_{indx}.png')

        try:
            utils.plot_embedding_end_points(embeddings, train_targs, embeds, algo_name=name,
                    wandb_logger=wandb_logger, plot_type='PHATE',
                    save_path=save_dir + f'max_fitness_PHATE_end_points_{indx}.png')

        except:
            print("phate end point plot failed")

        try:
            utils.plot_embedding_end_points(embeddings, train_targs, embeds, algo_name=name,
                        wandb_logger=wandb_logger, plot_type='TSNE',
                        save_path=save_dir + f'max_fitness_tsne_end_points_{indx}.png')
        except:
            print("tsne end point plot failed")


     # check embeddings
    # PCA
    emb_pca_coords = utils.plot_embedding(embeddings, train_targs,
                        wandb_logger=wandb_logger,
                        save_path=save_dir + 'original_fitness_lanscape_pca.png' )

    # utils.plot_landscape_3d(emb_pca_coords, train_targs, wandb_logger=wandb_logger,
    #                         save_path=save_dir + 'original_fitness_lanscape_pca_3D.png')


    # PHATE
    emb_phate_coords = utils.plot_embedding(embeddings, train_targs, plot_type='PHATE',
                        wandb_logger=wandb_logger,
                        save_path=save_dir + 'original_fitness_lanscape_phate.png' )

    # utils.plot_landscape_3d(emb_phate_coords, train_targs, wandb_logger=wandb_logger,
    #                          plot_type='PHATE',
    #                         save_path=save_dir + 'original_fitness_lanscape_phate_3D.png')


    # PHATE
    emb_TSNE_coords = utils.plot_embedding(embeddings, train_targs, plot_type='TSNE',
                        wandb_logger=wandb_logger,
                        save_path=save_dir + 'original_fitness_lanscape_tsne.png' )

    # utils.plot_landscape_3d(emb_TSNE_coords, train_targs, wandb_logger=wandb_logger,
    #                         plot_type='TSNE',
    #                         save_path=save_dir + 'original_fitness_lanscape_tsne_3D.png')






