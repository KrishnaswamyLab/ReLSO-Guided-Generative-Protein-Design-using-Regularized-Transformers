
# import unittest
# import datetime
# import os
# # import numpy as np
# import argparse

# import torch
# # from torch import nn, optim

# import pytorch_lightning as pl
# from pytorch_lightning import Trainer
# # from pytorch_lightning.loggers import WandbLogger
# # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# # from pytorch_lightning.core.lightning import LightningModule

# import relso.auxnetworks as auxnetworks
# import relso.models as hmodels
# import relso.data as hdata
# import relso.utils as hutils


# # --------------------------------------------

# model_types = [
#     'lstm', 
#     'seq2seq_aux',
#     'cnn',
#     'cnn_ae_aux',
#     'cnn_vae_aux',
#     'seq2seq_aux_vae',
#     'seq_transformer',
#     'seq2seq_lap_vae',
#     'cnn_vae_lap',
#                 ]
# sup_models = [
#     'cnn',
#     'lstm'
#     ]

# ae_models = [
#     'cnn_ae_aux',
#     'cnn_rae_aux',
#     'cnn_vae_aux',
#     'seq2seq_aux',
#     'seq2seq_aux_vae',
#     'seq2seq_lap_vae',
#     'cnn_vae_lap',
#     'seq2seq_aux_rae'
# ]

# transformer_models = [
#     'seq_transformer',
#     'transformer',
#     'transformer_rae',
#     'transformer_rae_cnn'
# ]

# auxnetworks = [
#     'base_reg',
#     'dropout_reg'
# ]


# BASE_HPARAMS = {
#             'input_dim': 22,
#             'dataset': 'gifford',
#             'task': 'recon',
#             'batch_size': 100, 
#             'log_dir': 'testing_logs',
#             'alpha_val': 0.5,
#             'beta_val': 0.005, 
#             'gamma_val': 1.0,
#             'eta_val': 0.01,
#             'sigma_val': 1.5,
#             'reg_ramp': False,
#             'vae_ramp': True,
#             'lr': 0.0001,
#             'n_epochs': 1,
#             'n_gpus': 0,
#             'seq_len': 0,
#             'embedding_dim': 40,
#             'bidirectional': True,
#             'auto_lr': False,
#             'kernel_size':4,
#             'latent_dim': 25,
#             'hidden_dim': 50,
#             'layers': 2,
#             'probs': 0.2,
#             'auxnetwork': 'base_reg'
#         }

# BASE_HPARAMS = argparse.Namespace(**BASE_HPARAMS)


# # --------------------------------------------


# def str2bool(v):
#     if isinstance(v, bool):
#        return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


# class ModelTesting(unittest.TestCase):

#     def setUp(self):


#         proto_data = hdata.str2data('gifford')
#         self.data = proto_data(dataset='gifford',
#                     task=BASE_HPARAMS.task,
#                     batch_size=BASE_HPARAMS.batch_size)

        
#     def test_ae_models(self):

#         for model_name in ae_models:
#             train_bool = self.train(model_name)

#             self.assertTrue(train_bool)

#     def test_sup_models(self):

#         for model_name in sup_models:
#             train_bool = self.train(model_name)

#             self.assertTrue(train_bool)

#     def test_transformer_models(self):

#         for model_name in transformer_models:
#             train_bool = self.train(model_name)

#             self.assertTrue(train_bool)


#     def train(self, model_name):

#         save_path = f'{BASE_HPARAMS.log_dir}/{model_name}/{BASE_HPARAMS.dataset}/'
         
#         print("\nnow tesing model: {}\n".format(model_name))
#         proto_model = hmodels.str2model(model_name)

#         BASE_HPARAMS.seq_len = self.data.seq_len

#         dev_model = proto_model(hparams=BASE_HPARAMS)

#         # most basic trainer, uses good defaults
#         print("quick dev run...")
#         trainer = pl.Trainer.from_argparse_args(BASE_HPARAMS,
#                                                 fast_dev_run=True)
                                                
#         trainer.fit(model=dev_model, 
#                     train_dataloader=self.data.train_dataloader(),
#                     val_dataloaders=self.data.valid_dataloader(), 
#                     )

#         print('\n dev run complete!\n')
   
#         print("\ntraining for 1 epoch\n")

#         tr_model = proto_model(hparams=BASE_HPARAMS)

#         trainer = pl.Trainer.from_argparse_args(BASE_HPARAMS,
#                                                 max_epochs=BASE_HPARAMS.n_epochs,
#                                                 default_root_dir=save_path)
                                                
#         trainer.fit(model=tr_model, 
#                     train_dataloader=self.data.train_dataloader(),
#                     val_dataloaders=self.data.valid_dataloader()
#         )

#         trainer.save_checkpoint(save_path + "/example.ckpt")
                            
#         print("finished training")

#         print("loading models")

#         loaded_model = proto_model.load_from_checkpoint(save_path + "/example.ckpt")

#         print("model loaded!")
#         print(loaded_model)


#         return True

# if __name__ == '__main__':
#     unittest.main()

