# ReLSO 

Improved Fitness Optimization Landscapes
for Sequence Design
- [Description](#Description)
- [Citation](#citation)
- [How to run   ](#how-to-run)
- [Training models](training-models)
- [Logging](#logging) 
- [Model Class](#model-class)
- [Optimizing in latent space](#optimizing-in-latent-space)
- [Original data source](#Original-data-sources)



## Description
---
In recent years, deep learning approaches for determining protein sequence-fitness
relationships have gained traction. Advances in high-throughput mutagenesis,
directed evolution, and next-generation sequencing have allowed for the accumulation of large amounts of labelled fitness data and consequently, attracted the
application of various deep learning methods. Although these methods learn an
implicit fitness landscape, there is little work on using the latent encoding directly
for protein sequence optimization. Here we show that this latent space representation of a fitness landscape can be made very amenable to latent space optimization
through a joint-training process. We also show that this encoding strategy which
also provides improvements to generalization over more traditional training strategies. We apply our approach to several biological contexts and show that latent
space optimization in a smooth learned folding landscape allows for more accurate
and efficient optimization of protein sequences.

## Citation

This repo accompanies the following publication:

*Egbert Castro, Abhinav Godavarthi, Julien Rubinfien, Smita Krishnaswamy. Guided Generative Protein Design using Regularized Transformers. Nature Machine Intelligence, in review (2021).*

## How to run   
---

First, install dependencies   
```bash
# clone project   
git clone https://github.com/ec1340/relso 


# install project   
cd relso 
pip install -e .   
pip install -r requirements.txt
 ```   

## Usage

### Training models
 
 ```bash
# run training script
python train_relso.py  --data gifford
```
---
*note: if arg option is not relevant to current model selection, it will not be used. See init method of each model to see what's used.

### available dataset args:

        gifford, AMIE_PSEAE, DLG_RAT, GB1_WU, RASH_HUMAN, RL401_YEAST, UBE4B_MOUSE, YAP1_HUMAN, GFP, TAPE (GFP by TAPE splits)

### available auxnetwork args:

        base_reg, dropout_reg

## Logging
---

By default, training logs are saved within 

        train_logs/<model>/<dataset>/<datetime>/

but can be set using 

        python train.py --dataset <dataset> --log_dir another_dir/

which would save the logs to:

       another_dir/<model>/<dataset>/<datetime>/

During each run, the following are saved:

- embeddings for train, val, and test set
- model weights (in the wandb dir)
- hyperparams (in the wandb dir)

## Model Class
---

To keep a consistent model forward call behavior, all models return the same pair of outputs

        outputs, z_rep = model(data)

- outputs: predictions made my 1 or more heads
- z_rep: latent representation


## Optimizing in latent space
---

To run the gradient-free optimization algorithms

        usage: run_optim.py [-h] --model MODEL --embeddings EMBEDDINGS --dataset DATASET --weights WEIGHTS [--n_steps N_STEPS] [--save_dir SAVE_DIR]

which will run the currently implemented algorithms and save plots of their optimization trajectories


        optional arguments:
        -h, --help            show this help message and exit
        --model MODEL
        --embeddings EMBEDDINGS
        --dataset DATASET
        --weights WEIGHTS
        --n_steps N_STEPS
        --save_dir SAVE_DIR


## Original data sources

- GIFFORD: 
