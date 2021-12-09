 
"""
Helper functions for data processing
"""
import numpy as np
import torch
import scipy

#-----------
# CoNSTANTS
# -----------

ENCODING = {"I":[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "L":[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "V":[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "F":[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "M":[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "C":[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "A":[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "G":[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "P":[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "T":[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "S":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "Y":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          "W":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          "Q":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          "N":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          "H":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          "E":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          "D":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          "K":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          "R":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          "X":[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,
               0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
          "J":[0, 0, 0, 0, 0, 0, 0, 0 ,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


SEQ2IND = {"I":0,
          "L":1,
          "V":2,
          "F":3,
          "M":4,
          "C":5,
          "A":6,
          "G":7,
          "P":8,
          "T":9,
          "S":10,
          "Y":11,
          "W":12,
          "Q":13,
          "N":14,
          "H":15,
          "E":16,
          "D":17,
          "K":18,
          "R":19,
          "X":20,
          "J":21}

IND2SEQ = {ind: AA for AA, ind in SEQ2IND.items()}


#-----------
# FUNCTIONS
# -----------

""" Get padded sequence from indices """
def inds_to_seq(seq):
    return [IND2SEQ[i] for i in seq]

""" Get indices of sequence """
def seq_to_inds(seq):
    return [SEQ2IND[i] for i in seq]

""" Get one-hot representation of padded sequence"""
def get_rep(seq):
    temp = torch.tensor([ENCODING[element] for element in seq])
    return torch.transpose(temp,0,1)

""" Get one-hot representation from indices of sequence"""
def get_rep_inds(seq):
    temp = torch.tensor([ENCODING[int(ind)] for ind in seq])
    return torch.transpose(temp,0,1)

""" Pad input sequence to length 20 on either side"""
def pad(seq):
    x = (20 - len(seq))/2
    temp = ['J']*int(np.floor(x))
    temp.extend(list(seq))
    temp.extend(['J']*int(np.ceil(x)))
    return ('').join(temp)

""" 
Load Ens_grad dataset
Args: inputted as a .csv/pandas DataFrame with sequences and targets
Returns: reps (converts sequence to indices) and targets (enrichment vals) as torch tensors
"""
def load_raw_giff_data(input_data):    
    targs = list(torch.from_numpy(np.array(input_data["enrichment"])))
    
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)
    reps = []
    seqs = list(input_data['CDR3'])
    for seq in seqs: 
        reps.append(torch.tensor(seq_to_inds(seq)))
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    
    return reps,targets


def load_raw_mut_data(input_data):
    targs = list(torch.from_numpy(np.array(input_data.iloc[:,1])))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    reps = []
    seqs = list(input_data.iloc[:,0])
    for seq in seqs: 
        reps.append(torch.tensor(seq_to_inds(seq)))
        
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    
    return reps,targets


def load_raw_happy_data(input_data, target_col, log_bool):
    targs = list(torch.from_numpy(input_data[target_col].to_numpy()))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)
    if log_bool:
        targets = torch.log(targets)

    reps = []
    seqs = list(input_data['cdr_sequence'])

    for seq in seqs: 
        
        # replace wildcard character
        seq = seq.replace('*', 'X')
        reps.append(torch.tensor(seq_to_inds(seq)))
        
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    
    return reps,targets
   
def load_raw_tape_data(input_data):
    targs = input_data.iloc[:,1].to_numpy()
    targs = list(torch.from_numpy(targs))
    targets = torch.cat([x.unsqueeze(dim=0).type("torch.FloatTensor") for x in targs], 0)

    reps = []
    seqs = list(input_data.iloc[:,0])
    for seq in seqs: 
        reps.append(torch.tensor(seq_to_inds(seq)))
                
    reps = torch.cat([x.type("torch.LongTensor").unsqueeze(dim=0) for x in reps])
    
    return reps,targets

def load_alphabet_dict(alphabet_loc):
    with open(alphabet_loc) as f: 
        alphabet = f.read().splitlines()

    alphabet = list(sorted(alphabet))
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    return symbol_to_idx


###############################
# FEATURIZING SEQUENCES
###############################


# def conv_set_