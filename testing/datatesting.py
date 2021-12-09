

import unittest
import relso.data as hdata
from relso import models

from argparse import ArgumentParser
import torch
import numpy as np

# ├── AMIE_PSEAE_test_data.csv
# ├── AMIE_PSEAE_train_data.csv
# ├── DLG_RAT_test_data.csv
# ├── DLG_RAT_train_data.csv
# ├── GB1_WU_test_data.csv
# ├── GB1_WU_train_data.csv
# ├── RASH_HUMAN_test_data.csv
# ├── RASH_HUMAN_train_data.csv
# ├── RL401_YEAST_test_data.csv
# ├── RL401_YEAST_train_data.csv
# ├── UBE4B_MOUSE_test_data.csv
# ├── UBE4B_MOUSE_train_data.csv
# ├── YAP1_HUMAN_test_data.csv
# └── YAP1_HUMAN_train_data.csv


class TestEnsDataLoading(unittest.TestCase):

    def setUp(self):
        self.recon_data = hdata.EnsGradData('./data/', task='recon')
        self.ns_data = hdata.EnsGradData('./data/', task='recon')

    def test_load(self):
        print("loading Gifford data")
        ens_recon_module = hdata.EnsGradData('./data/', task='recon')
        ens_ns_module = hdata.EnsGradData('./data/', task='next-step')

        data_setups = [ens_recon_module, ens_ns_module]
        
        print(f'\n Gifford seq len: {ens_recon_module.seq_len}\n')

        # check different task setups
        for dat in data_setups:
            self.assertTrue(dat.train_dataloader())
            self.assertTrue(dat.valid_dataloader())
            self.assertTrue(dat.test_dataloader())

    def test_is_tensor(self):
        
        test_tensor = torch.Tensor([0])

        self.assertIsInstance(self.recon_data.raw_train_tup[0], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_train_tup[1], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_test_tup[0], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_test_tup[0], type(test_tensor))

        self.assertIsInstance(self.ns_data.raw_train_tup[0], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_train_tup[1], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_test_tup[0], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_test_tup[0], type(test_tensor))

        self.assertIsInstance(self.recon_data.train_split_seqd, type(test_tensor))
        self.assertIsInstance(self.recon_data.valid_split_seqd, type(test_tensor))
        self.assertIsInstance(self.recon_data.test_split_seqd, type(test_tensor))
        

    def test_nans(self):
        '''
        check for nans
        '''
        tr_dat = self.recon_data.raw_train_tup[0]
        te_dat = self.recon_data.raw_test_tup[0]

        self.assertTrue(torch.isnan(tr_dat).sum().item() == 0)
        self.assertTrue(torch.isnan(te_dat).sum().item() == 0)
        


class TestMutDataLoading(unittest.TestCase):

    def setUp(self):
        self.recon_data = hdata.MutData('./data/', dataset='DLG_RAT', task='recon')
        self.ns_data = hdata.MutData('./data/', dataset='DLG_RAT', task='recon')

    def test_load(self):
        mut_datasets = ['AMIE_PSEAE',
                'DLG_RAT',
                'GB1_WU',
                'RASH_HUMAN',
                'RL401_YEAST',
                'UBE4B_MOUSE',
                'YAP1_HUMAN',
                'GFP'
                ]

        for indx, mut_dat in enumerate(mut_datasets):
            print(f'testing {mut_dat} data {indx+1}/{len(mut_datasets)}')

            ns_module = hdata.MutData('./data/', dataset=mut_dat, task='next-step')
            recon_module = hdata.MutData('./data/', dataset=mut_dat, task='recon')
            
            data_setups = [ns_module, recon_module]

            print(f'\n {mut_dat} seq len: {recon_module.seq_len}\n')
            
            # check different task setups
            for dat in data_setups:
                self.assertTrue(dat.train_dataloader())
                self.assertTrue(dat.valid_dataloader())

    def test_is_tensor(self):
        
        test_tensor = torch.Tensor([0])

        self.assertIsInstance(self.recon_data.raw_train_tup[0], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_train_tup[1], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_test_tup[0], type(test_tensor))
        self.assertIsInstance(self.recon_data.raw_test_tup[0], type(test_tensor))

        self.assertIsInstance(self.ns_data.raw_train_tup[0], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_train_tup[1], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_test_tup[0], type(test_tensor))
        self.assertIsInstance(self.ns_data.raw_test_tup[0], type(test_tensor))


        self.assertIsInstance(self.recon_data.train_split_seqd, type(test_tensor))
        self.assertIsInstance(self.recon_data.valid_split_seqd, type(test_tensor))
        self.assertIsInstance(self.recon_data.test_split_seqd, type(test_tensor))
        


    def test_nans(self):
        '''
        check for nans
        '''
        
        tr_dat = self.recon_data.raw_train_tup[0]
        te_dat = self.recon_data.raw_test_tup[0]

        self.assertTrue(torch.isnan(tr_dat).sum().item() == 0)
        self.assertTrue(torch.isnan(te_dat).sum().item() == 0)



if __name__ == '__main__':
    unittest.main()
