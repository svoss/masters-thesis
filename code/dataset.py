""" General dataset helpers """

from config import get_config
import os
import zipfile
import hashlib
import requests
import re
import numpy as np
import operator
from chainer import dataset
from chainer import datasets as D
from config import get_mongo_db,get_mongo_client


def create_alphabet(case_sensitive=True, digits=True, others="%.&[]()!~-+ ",do_unicode=False):
    """ Creates dictionary from character to integer index, also creates inverse
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if case_sensitive:
        alphabet += alphabet.upper()
    if digits:
        alphabet += "0123456789"
    alphabet += others
    if do_unicode:
        alphabet = unicode(alphabet)

    # 0 is catch all
    return dict([(a,i+1) for i,a in enumerate(alphabet)]), dict([(i+1,a) for i,a in enumerate(alphabet)])


def tokenize(line):
    line = line.replace("<br>", " ").replace(". ", " <eos> ").lower()
    for token in re.findall("[\w\<\>]+", line, re.UNICODE):
        yield token


def get_cache_folder():
    """ Gets folder to cache dataset too"""
    config = get_config()
    return os.path.join(config.get('folder', 'cache_prefix'),'.cache')


def extract_zip_into(from_file, to):
    """Extracts zip to certain directory"""
    with zipfile.ZipFile(from_file, "r") as z:
        z.extractall(to)


def download_file_from_url(url, to_path):
    """ Download file from url"""
    r = requests.get(url, stream=True)
    with open(to_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    return to_path


def download_if_not_exists(dataset_url):
    folder = os.path.join(get_cache_folder(), hashlib.md5(dataset_url).hexdigest())
    if not os.path.exists(folder):
        os.makedirs(folder)

    does_exists = len([fn for fn in os.listdir(folder) if fn[-5:] == '.json']) > 0
    if not does_exists:
        print "Dataset not available, downloading from %s" % dataset_url
        tmp_file = os.path.join(folder, 'data.zip')
        download_file_from_url(dataset_url, tmp_file)
        extract_zip_into(tmp_file, folder)
        os.remove(tmp_file)

    return folder


def find_chars(ingredient, text_size, max_text_size, char2int):
    chars = []
    for ci in xrange(max_text_size):
        if ci < text_size and ingredient[ci] in char2int:
            chars.append(char2int[ingredient[ci]])
        else:
            chars.append(0)  # padding and catch-all
    return chars


def map_word_sequence(input_string, inv_voc, counter=None):
    """ Tokenizes a string into sequence of ints, based on vocabulary
    Will also expand vocabulary if necessarily
    """
    seq = []
    for token in tokenize(input_string):
        if token not in inv_voc:
            inv_voc[token] = len(inv_voc)
        idx = inv_voc[token]
        seq.append(idx)
        if counter is not None:
            if idx not in counter:
                counter[idx] = 0
            counter[idx] += 1

    return seq


def map_to_new_voc(Y, mapping):
    """ Maps from old voc to new voc recursively in case you want to rewrite your vocabulary"""
    if type(Y) is list:
        return [x for x in [map_to_new_voc(y, mapping) for y in Y] if x is not None]
    elif Y in mapping:
        return mapping[Y]
    else:
        None


# Shared helping functions
def create_normal_voc(voc):
    """ Creates inverse vocabulary from word to index
    """
    l = ['' for v in xrange(len(voc))]
    for w, i in voc.iteritems():
        l[i] = w
    return l


def apply_threshold_to_voc(Y, inv_voc, voc_counter, min_th):
    """Maps all words that have less occurrences in Y according to voc_counter below min_th to a single below threshold
    While rewriting all other words to prevent holes
    """
    voc = create_normal_voc(inv_voc)

    # create new mapping going from old idx to new
    new_mapping = {}
    new_voc = {}  # also build new vocubulary
    for x, y in voc_counter.iteritems():
        if y >= min_th:
            idx = len(new_mapping)
            new_mapping[x] = idx
            new_voc[voc[x]] = idx

    # then walk over sequences:
    return map_to_new_voc(Y, new_mapping), new_voc


def convert_to_count_array(sequences, dtype=np.int8, max_value=None):
    """ Gets a list of sequences of integers,
    """
    if max_value is None:
        max_value = max([max(s) for s in sequences if len(s) > 0])
    X = np.zeros((len(sequences), max_value + 1), dtype=dtype)
    for idx in xrange(len(sequences)):
        sequence = sequences[idx]
        for s in sequence:
            X[idx, s] += 1
    return X

def create_sub_dataset(X_chars, Y_old, cond=lambda y: True,new_y=lambda y: y):
    """ Creates dataset when y values corresponds to certain requirement"""
    Y = []
    X = []
    for idx in xrange(len(Y_old)):
        y = Y_old[idx]
        if cond(y):
            Y.append(new_y(y))
            X.append(X_chars[idx])
    return np.array(X,dtype=np.int8),np.array(Y, dtype=np.int8)

def print_recipe(X_chars, alphabet):
    if len(X_chars.shape) != 2:
        raise Exception("Recipe should have 2 dimensions")
    for idx in xrange(X_chars.shape[0]):
        t = "".join([alphabet[X_chars[idx,i]] for i in xrange(X_chars.shape[1])])
        if t != "":
            print "- %s" % t

def get_recipe_array(X_chars, alphabet):
    """One hot encoded version"""
    if len(X_chars.shape) != 3:
        raise Exception("Recipe should have 3 dimensions")

    idx = np.argwhere(X_chars)
    ingredients = [["" for x in xrange(X_chars.shape[2])] for i in xrange(X_chars.shape[1])]
    for ri in xrange(idx.shape[0]):
        char, ingredient, pos = idx[ri, :]
        ingredients[ingredient][pos] += alphabet[char + 1]
    return ["".join(ing) for ing in ingredients]

def print_recipe_oe(X_chars, alphabet):
    ingredients = get_recipe_array(X_chars, alphabet)
    print "\n".join(["- "+x for x in ingredients  if x != "" ])


def split_dataset(dataset, train_frac=.8, test_frac=.5):
    train_end = int(len(dataset) * train_frac)
    train, rest = D.split_dataset(dataset, train_end)
    test_end = int(len(rest) * test_frac)
    test, val = D.split_dataset(rest, test_end)
    return train, test, val


def make_dataset(X,Y,a_size,embed=False):
    if embed:
        dataset = D.TupleDataset(X.astype(np.int32), Y)
    else:
        dataset = D.TupleDataset(OneHotEncodingDataset(X, a_size), Y)
    return dataset


class MongoDBDataset(dataset.DatasetMixin):
    """ Dataset type streams documents from mongodb query
    """
    def __init__(self, collection, ids):
        self.ids = ids
        self.collection = collection

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        If index is slice will return N+1 dimensions, if scalar will return N dimensions. Second dimension will be vector_size
        :param index: 
        :return: 
        """

        if isinstance(index,slice):
            return self.collection.find({'_id':{"$in": self.ids[index]}})
        else:
            return self.collection.find_one({'_id':self.ids[index]})


def build_mongo_datasets(collection, query,val_frac=.1, test_frac=.1,limit_to=None):
    q = collection.find(query,{"_id":1})
    ids = []
    for d in q:
        ids.append(d['_id'])
        if limit_to is not None:
            limit_to -= 1
            if limit_to < 1:
                break
    start_val = int((1.-(val_frac + test_frac)) * len(ids))
    start_test = int((1.-test_frac) * len(ids))

    train, val, test = ids[0:start_val],ids[start_val:start_test],ids[start_test:-1]
    return train,val,test

class OneHotEncodingDataset(dataset.DatasetMixin):
    """ Class that converts integers to one-hot-encoded vectors. 0's will be encoded as all 0's"""
    def __init__(self, X, vector_size, dtype=np.float32):
        """
        :param X: N-dimesional dataset of integer values
        :param vector_size: alphabet size of(The max integer found in X)
        """
        self.X = X
        self.length = len(X)
        self.vector_size = vector_size
        self.dtype = dtype

    def __getitem__(self, index):
        """
        If index is slice will return N+1 dimensions, if scalar will return N dimensions. Second dimension will be vector_size
        :param index: 
        :return: 
        """
        if isinstance(index,slice):
            return self._one_hot_encode(self.X[index])
        else:
            return self._one_hot_encode(self.X[index],roll_to=0)

    def _one_hot_encode(self, X,roll_to=1):
        shape = list(X.shape)

        l = reduce(operator.mul, shape, 1) # Total length if was list of ints
        X = X.reshape((l,)) # reshape it to list of ints
        encoded = np.zeros((l, self.vector_size + 1),self.dtype) # Build matrix of l x vector_size + 1
        encoded[np.arange(l), X] = 1 # One on the right places
        encoded = encoded[:, 1:] # remove first row, 0 = all zero's
        encoded = encoded.reshape(tuple(shape + [self.vector_size])) # reshape back
        encoded = np.rollaxis(encoded, len(shape), roll_to)
        return encoded

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass