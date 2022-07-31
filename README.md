
<div align="center">    
 
# ReLSO: A Transformer-based Model for Latent Space Optimization and Generation of Proteins
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2201.09948)
-->

[![Paper](https://img.shields.io/badge/arxiv-2006.06885-B31B1B.svg)](https://arxiv.org/abs/2201.09948)

[![DOI](https://zenodo.org/badge/436740631.svg)](https://zenodo.org/badge/latestdoi/436740631)
  
</div>

Improved Fitness Optimization Landscapes
for Sequence Design
- [Description](#Description)
- [Citation](#citation)
- [How to run   ](#how-to-run)
- [Training models](training-models)
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
git clone https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers.git


# install requirements

# with conda
conda env create -f relso_env.yml

# with pip
pip install -r requirements.txt

# install relso
cd ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers 
pip install -e .   
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

        gifford, GB1_WU, GFP, TAPE

### available auxnetwork args:

        base_reg




### Running optimization algorithms 
 
 ```bash
python run_optim.py --weights <path to ckpt file>/model_state.ckpt --embeddings  <path to embeddings file>train_embeddings.npy --dataset gifford
```
---



## Original data sources

- GIFFORD: https://github.com/gifford-lab/antibody-2019/tree/master/data/training%20data
- GB1: https://elifesciences.org/articles/16965#data
- GFP: https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154
- TAPE: https://github.com/songlab-cal/tape#data

