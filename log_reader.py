import argparse
import os
import json

def find_data_key(fn, data_key):
    if not os.path.exists(fn):
        return False
    with open(fn,'r') as io:
        X = json.load(io)
    return X[data_key]

def find_max_val_acc(fn, log_key):
    if not os.path.exists(fn):
        return False
    with open(fn,'r') as io:
        X = json.load(io)
    vals = [x[log_key] for x in X if log_key in x]
    if len(vals) > 0:
        return max(vals)
    else:
        return 0.0

def get_data(fn, data_file, key):
    if data_file:
        return find_data_key(fn, key)
    else:
        return find_max_val_acc(fn, key)


def search_in(dir, data_file, key, prefix=""):

    filename = 'log.json' if data_file else 'log'

    for d in os.listdir(dir):
        fd = os.path.join(dir, d)
        if os.path.isdir(fd):
            search_in(fd, data_file, key,"%s-%s" % (prefix, d))
        elif os.path.isfile(fd) and d == filename:
            print "%s: %.3f" % (prefix, get_data(fd, data_file, key))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find best accuracy in script')
    parser.add_argument('loc', type=str)
    parser.add_argument('--data-file', action='store_true')
    parser.add_argument('--folder', action='store_true')
    parser.add_argument('--key',type=str,default='validation/main/accuracy')
    args = parser.parse_args()
    if not args.folder:
        print get_data(args.loc, args.data_file, args.key)
    else:
        search_in(args.loc, args.data_file, args.key)