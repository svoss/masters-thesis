import matplotlib
# This makes plots work via ssh
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import sys
from os.path import dirname,realpath
import os
path = os.path.join(dirname(dirname(realpath(__file__))), '../code')
sys.path.append(path)
from preliminary_dataset import get_dataset
from dataset import  OneHotEncodingDataset, split_dataset
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.datasets as D
import chainer.iterators as I
import chainer.optimizers as O
import chainer.training as T
import chainer.training.extensions as E
import numpy as np
from model import build_model
import time
import argparse
from config import get_config,load_ex_controller
load_ex_controller()
from ec.manager import EM
from ec.model import Experiment


def get_args():
    parser = argparse.ArgumentParser(description='Run the preliminary experiment')
    parser.add_argument('--gpu',type=int,default=-1,help="GPU device to use")
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--lr',type=float,default='.01')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--momentum',type=float,default=.9)
    parser.add_argument('--case-sensitive',action='store_true')
    parser.add_argument('--num-chars',type=int, default=128)
    parser.add_argument('--track-name',type=str,default=None)
    parser.add_argument('--take-model-snapshot',action='store_true')
    parser.add_argument('--ds-parts',type=str,default='cuisine')
    return parser.parse_args()

def train_model(model, X, Y, alphabet, exp):

    out = os.path.join(get_config().get("folder", "results_prefix"), exp.startDate.strftime('%Y-%m-%d'), exp.id)
    dataset = D.TupleDataset(OneHotEncodingDataset(X, len(alphabet)), Y)
    # Make train, test and val set
    train, test, val = split_dataset(dataset)
    print len(train), len(test), len(val)

    # We make use of the multi process iteration such that the one-hot-encoding is done on a separate thread from the
    # GPU controller
    train_iter = I.MultiprocessIterator(train, exp.args['batch_size'])
    test_iter = I.MultiprocessIterator(test, exp.args['batch_size'], shuffle=False)
    val_iter = I.MultiprocessIterator(val, exp.args['batch_size'], shuffle=False, repeat=False)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")

    fn_a = 'loss_%s.png' % date
    fn_b = 'lr_%s.png' % date
    fn_c = 'acc_%s.png' % date
    loss_r = E.PlotReport(['validation/main/loss', 'main/loss'], 'epoch', file_name=fn_a)
    lr_r = E.PlotReport(['lr'], 'epoch', file_name=fn_b)
    acc_r = E.PlotReport(['validation/main/accuracy'], 'epoch', file_name=fn_c)

    classifier = L.Classifier(model)
    classifier.compute_accuracy = False # only probabilities for training
    if exp.args['gpu'] >= 0:
        classifier.to_gpu(exp.args['gpu'])
    print "Model intialized, epochs: %d" % exp.args['epochs']
    optimizer = O.MomentumSGD(lr=exp.args['lr'], momentum=exp.args['momentum'])
    optimizer.setup(classifier)

    # prepare model for evaluation
    eval_model = classifier.copy()
    eval_model.compute_accuracy = True
    eval_model.predictor.train = False

    updater = T.StandardUpdater(train_iter, optimizer, device=exp.args['gpu'])
    trainer = T.Trainer(updater, (exp.args['epochs'], 'epoch'), out)

    val_interval = (10 if exp.args['test'] else 1000), 'iteration'
    log_interval = (10 if exp.args['test'] else 1000), 'iteration'

    trainer.extend(E.Evaluator(
        val_iter, eval_model, device=exp.args['gpu']))
    trainer.extend(E.dump_graph('main/loss'))
    trainer.extend(E.ExponentialShift('lr', 0.5),
                   trigger=(7, 'epoch'))
    #trainer.extend(E.snapshot(), trigger=val_interval)
    if exp.args['take_model_snapshot']:
        trainer.extend(E.snapshot_object(
            model, 'model'), trigger=(exp.args['epochs'],'epoch'))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(E.LogReport(trigger=log_interval))
    trainer.extend(E.observe_lr(), trigger=log_interval)
    trainer.extend(E.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(loss_r)
    trainer.extend(lr_r)
    trainer.extend(acc_r)

    trainer.extend(T.extensions.ProgressBar(update_interval=10))

    trainer.run()


def train(exp):
    n_chars = exp.args['num_chars']
    case_sensitive = exp.args['case_sensitive']
    X,Y,alphabet = get_dataset(n_chars, case_sensitive, exp.args['ds_parts'])
    print "Alphabet consists of %d characters" % len(alphabet)
    #from collections import Counter
   # for x,c in Counter(Y).iteritems():
     #   print "%d: %d" % (x,c)

    # test_one_hot_encoding(X, alphabet)

    model = build_model(alphabet, np.max(Y) + 1, recipe_type='max')
    start = time.time()
    train_model(model, X, Y, alphabet, exp)

def test_one_hot_encoding(X, alphabet):
    from dataset import print_recipe, print_recipe_oe
    dataset = OneHotEncodingDataset(X, len(alphabet), dtype=np.int8)
    print_recipe(X[0], alphabet)
    print "-------"
    print_recipe_oe(dataset[0], alphabet)


if __name__ == "__main__":
    args = get_args()
    exp = Experiment(__file__, vars(args),args.track_name)
    if exp.do_track:
        try:
            EM.save(exp)
            train(exp)

            if exp is not None:
                exp.success()
                EM.update(exp)
        except Exception as e:
            exp.register_error(str(e))
            EM.update(exp)
            raise e
    else:
        train(exp)
    EM.try_move_output(exp)



