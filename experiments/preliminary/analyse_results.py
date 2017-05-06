"""
This script is used to 
"""
import sys
from os.path import dirname,realpath
import os
path = os.path.join(dirname(dirname(realpath(__file__))), '../code')
sys.path.append(path)
import argparse
from preliminary_dataset import get_dataset
from model import build_model
import numpy as np
from dataset import get_recipe_array, OneHotEncodingDataset
from dataset import get_cache_folder
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu',type=int,default=-1,help="GPU device to use")
    parser.add_argument('--model-params',type=str)
    parser.add_argument('--ds-parts', type=str, default='cuisine')
    parser.add_argument('--num-chars',type=int, default=128)
    parser.add_argument('--case-sensitive', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    tmp = os.path.join(get_cache_folder(), 'tmp.npz')
    if os.path.exists(tmp):
        data = np.load(tmp)
        X = data['X']
        Y = data['Y']
        alphabet = data['alphabet']
    else:
        X, Y, alphabet = get_dataset(args.num_chars, args.case_sensitive, args.ds_parts)
        X = X[:2]
        Y = Y[:2]
        np.savez(tmp, X=X, Y=Y, alphabet=alphabet)

    dataset = OneHotEncodingDataset(X, len(alphabet))
    model = build_model(alphabet, np.max(Y) + 1, recipe_type='max', load_file=args.model_params)
    A = model.ingredient(dataset[:2])
    A = np.rollaxis(A.data,2,1).reshape(2,32,256*6)
    print A.shape
    ings = get_recipe_array(dataset[:2][0],alphabet)
    for a in np.arange(0,32):
        print a,ings[a], np.sum(A[0,a])
        if ings[a] == '':
            break
    from collections import Counter
    MAX = np.argmax(A, 1)
    print Counter(MAX[0])






