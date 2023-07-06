import json
import pathlib
from pathlib import Path

def weight_path_from_json(json_path, weight_dir, dataset, model):
    """
    constructs weight path from info
    """

    with open(json_path) as f:
        model_dict = json.load(f)


    base_path = Path(str(weight_dir))
    
    
    timestamp = model_dict[str(dataset)][str(model)]
    fname = f'{timestamp}_model_weights.ckpt'

    return str(base_path.joinpath(fname))


def embed_path_from_json(json_path, embed_dir, dataset, model):
    """
    constructs weight path from info
    """

    with open(json_path) as f:
        model_dict = json.load(f)


    base_path = Path(str(embed_dir))
    
    
    timestamp = model_dict[str(dataset)][str(model)]

    train_fname = f'{timestamp}_train-embeddings.npy'
    test_fname = f'{timestamp}_test-embeddings.npy'

    train_fname = str(base_path.joinpath(train_fname))
    test_fname = str(base_path.joinpath(test_fname)) 

    return (train_fname, test_fname) 