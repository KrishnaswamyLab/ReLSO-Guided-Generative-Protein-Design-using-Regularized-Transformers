
<div align="center">    
 
# ReLSO: A Transformer-based Model for Latent Space Optimization and Generation of Proteins
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.2006.06885.svg)](https://arxiv.org/abs/2201.09948)
-->

[![Paper](https://img.shields.io/badge/arxiv-2006.06885-B31B1B.svg)](https://arxiv.org/abs/2201.09948)

[![Nature Machine Intelligence](https://img.shields.io/badge/Nature%20Machine%20Intelligence-2022-<COLOR>.svg)](https://www.nature.com/articles/s42256-022-00532-1)

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

Castro, Egbert, Abhinav Godavarthi, Julian Rubinfien, Kevin Givechian, Dhananjay Bhaskar, and Smita Krishnaswamy. "Transformer-based protein generation with regularized latent space optimization." Nature Machine Intelligence 4, no. 10 (2022): 840-851.



## Setup
---

### 1. Clone project
```bash
# clone project   
git clone https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers.git
```

### 2. Install PyTorch

see [PyTorch Installation page](https://pytorch.org/get-started/locally/) for more details. For convenience, here are some common options

```bash
# make conda environment
conda create --name relsoenv python=3.9
conda activate relsoenv

# install pytorch
# GPU (linux)
pip3 install torch torchvision torchaudio

# CPU only (linux)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CPU only (mac)
pip3 install torch torchvision torchaudio
```

### 3. Install other dependencies
```bash
python -m pip install networkx pytorch-lightning==1.9 wandb scikit-learn pandas matplotlib gdown phate
```   

### 4. Install ReLSO

```bash  
pip install -e .   
```   




## Usage

### Training models
 
 ```bash
# GPU training
python train_relso.py  --data gifford

# CPU training
python train_relso.py  --data gifford --cpu
```


---
*note: if arg option is not relevant to current model selection, it will not be used. See init method of each model to see what's used.

### available dataset args:

        gifford, GB1_WU, GFP, TAPE

### available auxnetwork args:

        base_reg


### Downloading Trained Models

 ```bash
bash download_weights.sh
```

which will create a directory called `relso_model_weights`

```
❯ tree relso_model_weights -L 1
relso_model_weights
├── model_embeddings
├── model_embeddings.zip
├── trained_models
├── trained_models.json
└── trained_models.zip

2 directories, 3 files
```



#### Examples
1. Loading GIFFORD Model


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

