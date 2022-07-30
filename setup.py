#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='relso',
      version='0.0.1',
      description='Regularized latent space optimization for protein sequences',
      author='Krishnaswamy Lab',
      url='https://github.com/KrishnaswamyLab/ReLSO-Guided-Generative-Protein-Design-using-Regularized-Transformers',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
      install_requires=[
            'numpy',
             'torch',
             'pytorch-lightning',
            'wandb',
            'sklearn',
            'matplotlib',
            'networkx'],

      packages=find_packages()
      )
