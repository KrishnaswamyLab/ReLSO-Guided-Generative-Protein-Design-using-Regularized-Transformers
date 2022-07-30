
"""
for loading data into model-ready dataloaders
"""

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

from relso.utils import data_utils
from scipy.spatial.distance import cdist
# import selfies as sf

# ---------------------
# CONSTANTS
# ---------------------

ROOT_DATA_DIR = './data/'
# ENS_GRAD_TRAIN = ENS_GRAD_DIR + 'train_data.csv'
# ENS_GRAD_TEST = ENS_GRAD_DIR + 'test_data.csv'

MUT_GRAD_DIR = './data/mut_data/'
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

MUT_DATASETS = ['GB1_WU',
                'GFP']


MUT_SEQUENCES = {
    'GFP': 'SKGEELFTGVVPILVELDGDVNGHKFNVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFESAMPEGHVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDYKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQDTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLESVTAAGITHGMDELYK',
    'Gifford': 'JJJJAAAAYDYWFDYJJJJJ',
    'GB1_WU': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'}


SEQ_DISTANCES_DIR = './data/seq_distances/'

# -------------------
# DATA MODULES
# -------------------
class EnsGradData(pl.LightningDataModule):
    """
    Gifford dataset
    """
    def __init__(self, data_dir=ROOT_DATA_DIR,
                        dataset=None,
                        task='recon',
                        train_val_split=0.7,
                        batch_size=100,
                        seqdist_cutoff=None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading data from: {self.data_dir}')

        print(f'setting up seq distances')
        self._setup_distances()
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()


    def _prepare_data(self):

        # load files
        train_df = pd.read_csv(str(self.data_dir + 'gifford_data/train_data.csv'))
        test_df = pd.read_csv(str(self.data_dir + 'gifford_data/test_data.csv'))

        if self.seqdist_cutoff:

            print(f'sequence distance split chosen with distance = {self.seqdist_cutoff}')
            full_data_df = pd.concat((train_df, test_df), 0)
            full_seqdist = np.concatenate((self.train_seq_dist.reshape(-1,1),
                                            self.test_seq_dist.reshape(-1,1)), 0).reshape(-1).astype(int)

            print(f'seq dist stats:\nmin: {full_seqdist.min()}\nmax: {full_seqdist.max()} \nmean: {full_seqdist.mean()}')

            assert len(full_data_df) == len(full_seqdist), 'data dfs and seq distance arrays do no match'

            dataset_indx = np.arange(len(full_seqdist))
            below_cutoff_indx = dataset_indx[full_seqdist <= int(self.seqdist_cutoff)]
            above_cutoff_indx = dataset_indx[full_seqdist > int(self.seqdist_cutoff)]

            train_df = full_data_df.iloc[below_cutoff_indx]
            test_df = full_data_df.iloc[above_cutoff_indx]

            self.train_seq_dist = full_seqdist[below_cutoff_indx]
            self.test_seq_dist = full_seqdist[above_cutoff_indx]


        train_seqs, train_fitness = data_utils.load_raw_giff_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_giff_data(test_df)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')


    def _setup_distances(self):

        # starts at 1 since header is included in seq distances :(
        self.train_seq_dist = torch.from_numpy(np.loadtxt(self.data_dir + 'seq_distances/gifford_train_data_seq_distances.csv')[1:]).reshape(-1)
        self.test_seq_dist = torch.from_numpy(np.loadtxt(self.data_dir + 'seq_distances/gifford_test_data_seq_distances.csv') [1:]).reshape(-1)

        print(f'seq distances shapes: {self.train_seq_dist.shape}\t{self.test_seq_dist.shape}')

    def _setup_task(self):

        if self.task == 'next-step':

            # train set
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)

            # test set
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')

        train_all_data = [train_data, train_targets, train_fitness, self.train_seq_dist]
        train_all_data_numpy = [x.numpy() for x in train_all_data]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_seqd, v_seqd = train_test_split(*train_all_data_numpy,
                                                                    train_size=train_size,
                                                                    random_state=42)

        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_seqd]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_seqd]]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list[:-1])
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list[:-1])

        # self.train_dataset = torch.utils.data.TensorDataset(t_dat, t_tar, t_fit)
        # self.valid_dataset = torch.utils.data.TensorDataset(v_dat, v_tar, v_fit)

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)

        self.train_split_seqd = t_seqd
        self.valid_split_seqd = v_seqd
        self.test_split_seqd = self.test_seq_dist

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = False)


# ----------------------------
# Mutational classes
# ----------------------------

class MutData(pl.LightningDataModule):
    def __init__(self, data_dir=ROOT_DATA_DIR,
                                dataset='AMIE_PSEAE',
                                task='recon',
                                train_val_split=0.7,
                                batch_size=100,
                                seqdist_cutoff=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        self.train_data_file = '{}_train_data.csv'.format(dataset)
        self.test_data_file = '{}_test_data.csv'.format(dataset)

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        print(f'setting up seq distances')
        self._setup_distances()
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()


    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.train_data_file), header=None)
        test_df = pd.read_csv(str(self.data_dir + 'mut_data/' + self.test_data_file),  header=None)

        if self.seqdist_cutoff:
            print(f'sequence distance split chosen with distance = {self.seqdist_cutoff}')
            full_data_df = pd.concat((train_df, test_df), 0)
            full_seqdist = np.concatenate((self.train_seq_dist.reshape(-1,1),
                                            self.test_seq_dist.reshape(-1,1)), 0).reshape(-1).astype(int)

            print(f'seq dist stats:\nmin: {full_seqdist.min()}\nmax: {full_seqdist.max()} \nmean: {full_seqdist.mean()}')

            assert len(full_data_df) == len(full_seqdist), 'data dfs and seq distance arrays do no match'

            dataset_indx = np.arange(len(full_seqdist))

            below_cutoff_indx = dataset_indx[full_seqdist <= int(self.seqdist_cutoff)]
            above_cutoff_indx = dataset_indx[full_seqdist > int(self.seqdist_cutoff)]

            train_df = full_data_df.iloc[below_cutoff_indx]
            test_df = full_data_df.iloc[above_cutoff_indx]

            self.train_seq_dist = full_seqdist[below_cutoff_indx]
            self.test_seq_dist = full_seqdist[above_cutoff_indx]


        train_seqs, train_fitness = data_utils.load_raw_mut_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_mut_data(test_df)

        # log scaling fitness
        # train_fitness += train_fitness.min()
        # test_fitness += test_fitness.min()

        # train_fitness, test_fitness = torch.log(train_fitness), torch.log(test_fitness)

        #train_fitness = train_fitness + torch.abs(train_fitness.min()) + 0.001
        #test_fitness = test_fitness + torch.abs(test_fitness.min()) + 0.001

        #train_fitness = torch.log(train_fitness)
        #test_fitness = torch.log(test_fitness)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test sizes: {(self.train_N, self.test_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_distances(self):

        self.train_seq_dist = torch.from_numpy(np.loadtxt(self.data_dir + 'seq_distances/' +  f'{self.dataset}_train_data_seq_distances.csv')).reshape(-1)
        self.test_seq_dist = torch.from_numpy(np.loadtxt(self.data_dir + 'seq_distances/' + f'{self.dataset}_test_data_seq_distances.csv') ).reshape(-1)

        print(f'seq distances shapes: {self.train_seq_dist.shape}\t{self.test_seq_dist.shape}')

    def _setup_task(self):

        if self.task == 'next-step':
            # LSTM training dataset (next-char prediction)
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]

        train_size = int(self.train_N * 0.85)
        val_size = self.train_N - train_size
        print(f'split sizes\ntrain: {train_size}\nvalid: {val_size}')


        train_all_data = [train_data, train_targets, train_fitness, self.train_seq_dist]
        train_all_data_numpy = [x.numpy() for x in train_all_data ]

        t_dat, v_dat, t_tar, v_tar, t_fit, v_fit, t_seqd, v_seqd = train_test_split(*train_all_data_numpy,
                                                                    train_size=train_size,
                                                                    random_state=42)


        train_split_list = [torch.from_numpy(x) for x in [t_dat, t_tar, t_fit, t_seqd]]
        valid_split_list = [torch.from_numpy(x) for x in [v_dat, v_tar, v_fit, v_seqd]]

        # train_split_list = [t_dat, t_tar, t_fit, t_seqd]
        # valid_split_list = [v_dat, v_tar, v_fit, v_seqd]

        self.train_dataset = torch.utils.data.TensorDataset(*train_split_list[:-1])
        self.valid_dataset = torch.utils.data.TensorDataset(*valid_split_list[:-1])

        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)

        self.train_split_seqd = t_seqd
        self.valid_split_seqd = v_seqd
        self.test_split_seqd = self.test_seq_dist


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = False)


# ----------------------------
# TAPE data class
# ----------------------------

class TAPE(pl.LightningDataModule):
    def __init__(self, data_dir=ROOT_DATA_DIR,
                                dataset='TAPE',
                                task='next-step',
                                train_val_split=0.7,
                                batch_size=100,
                                seqdist_cutoff=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqdist_cutoff = seqdist_cutoff

        self.train_data_file = '{}_train_data.csv'.format(dataset)
        self.test_data_file = '{}_test_data.csv'.format(dataset)
        self.valid_data_file = '{}_valid_data.csv'.format(dataset)

        # check task
        assert task in ['recon', 'next-step']
        self.task = task

        print(f'loading dataset: {self.dataset} from: {self.data_dir}')
        print(f'setting up seq distances')
        self._setup_distances()
        self._prepare_data()

        print(f'setting up task: {self.task}')
        self._setup_task()

        print('setting up splits')
        self.setup()

    def _prepare_data(self):
        # load files
        train_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.train_data_file))
        test_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.test_data_file))
        valid_df = pd.read_csv(str(self.data_dir + '/mut_data/' + self.valid_data_file))


        train_seqs, train_fitness = data_utils.load_raw_tape_data(train_df)
        test_seqs, test_fitness = data_utils.load_raw_tape_data(test_df)
        valid_seqs, valid_fitness = data_utils.load_raw_tape_data(valid_df)

        self.raw_train_tup = (train_seqs, train_fitness)
        self.raw_test_tup = (test_seqs, test_fitness)
        self.raw_valid_tup = (valid_seqs, valid_fitness)

        self.train_N = train_fitness.shape[0]
        self.test_N = test_fitness.shape[0]
        self.valid_N = valid_fitness.shape[0]

        self.seq_len = train_seqs.shape[1]

        print(f'train/test/valid sizes: {(self.train_N, self.test_N, self.valid_N)}')
        print(f'seq len: {self.seq_len}')

    def _setup_distances(self):

        # starts at 1 since header is included in seq distances :(
        self.train_seq_dist = np.loadtxt(self.data_dir + '/seq_distances/TAPE_train_data_seq_distances.csv')
        self.test_seq_dist = np.loadtxt(self.data_dir + '/seq_distances/TAPE_test_data_seq_distances.csv')
        self.val_seq_dist = np.loadtxt(self.data_dir + '/seq_distances/TAPE_valid_data_seq_distances.csv')

        print(f'seq distances shapes: {self.train_seq_dist.shape}\t{self.test_seq_dist.shape}')

    def _setup_task(self):

        if self.task == 'next-step':
            # LSTM training dataset (next-char prediction)
            train_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_train_tup[0]],dim=0)
            train_targets = torch.stack([rep[1:] for rep in self.raw_train_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            test_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_test_tup[0]],dim=0)
            test_targets = torch.stack([rep[1:] for rep in self.raw_test_tup[0]],dim=0)


            # LSTM training dataset (next-char prediction)
            valid_data = torch.stack([rep[:self.seq_len-1] for rep in self.raw_valid_tup[0]],dim=0)
            valid_targets = torch.stack([rep[1:] for rep in self.raw_valid_tup[0]],dim=0)

        elif self.task == 'recon':
            # reconstruction
            train_data = self.raw_train_tup[0]
            train_targets = self.raw_train_tup[0]

            # reconstruction
            test_data = self.raw_test_tup[0]
            test_targets = self.raw_test_tup[0]

            # reconstruction
            valid_data = self.raw_valid_tup[0]
            valid_targets = self.raw_valid_tup[0]


        train_fitness = self.raw_train_tup[1]
        test_fitness = self.raw_test_tup[1]
        valid_fitness = self.raw_valid_tup[1]

        self.train_dataset = torch.utils.data.TensorDataset(train_data, train_targets, train_fitness)
        self.test_dataset = torch.utils.data.TensorDataset(test_data, test_targets, test_fitness)
        self.valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_targets, valid_fitness)

        self.train_split_seqd = self.train_seq_dist
        self.valid_split_seqd = self.val_seq_dist
        self.test_split_seqd = self.test_seq_dist

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        # split into train and validation splits
        print("setting up train and validation splits")
        self.train_split, self.valid_split =  self.train_dataset, self.valid_dataset

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        print ("setting up test split")
        self.test_split = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, shuffle = True)

    def valid_dataloader(self):
        return DataLoader(self.valid_split, batch_size=self.batch_size, shuffle = False)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, shuffle = False)


def str2data(dataset_name):
    """returns an uninitialized data module

    Args:
        arg ([type]): [description]

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    # model dict

    if dataset_name == 'gifford':
        data = EnsGradData

    elif dataset_name in MUT_DATASETS:
        data = MutData

    elif dataset_name == 'TAPE':
        data = TAPE

    else:
        raise NotImplementedError(f'{dataset_name} not implemented')

    return data
