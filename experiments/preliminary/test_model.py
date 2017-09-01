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
from dataset import  make_dataset, split_dataset
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
from datetime import datetime
import time
import argparse
import json
from config import get_config
from links import MultiTaskClassifier


def get_args():
    parser = argparse.ArgumentParser(description='Run the preliminary experiment')
    parser.add_argument('--gpu',type=int,default=-1,help="GPU device to use")
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--lr',type=float,default='.01')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--momentum',type=float,default=.0)
    parser.add_argument('--case-sensitive',action='store_true')
    parser.add_argument('--num-chars',type=int, default=128)
    parser.add_argument('--val-chars', type=int,default=None)
    parser.add_argument('--track-name',type=str,default=None)
    parser.add_argument('--take-model-snapshot',action='store_true')
    parser.add_argument('--ds-parts',type=str,default='cuisine')
    parser.add_argument('--recipe-type',type=str,default='max')
    parser.add_argument('--ingredient-type', type=str, default='wide')
    parser.add_argument('--acc-val',action='store_true')
    parser.add_argument('--acc-train', action='store_true')
    parser.add_argument('--no-validation',action='store_false')
    parser.add_argument('--embed', action='store_true')
    parser.add_argument('--num-ingredients', type=int, default=32)
    parser.add_argument('--val-ingredients', type=int,default=16)
    parser.add_argument('--cus-factor', type=float, default=.5)
    parser.add_argument('--veg-factor', type=float, default=.5)
    return parser.parse_args()

def train_model(model, X, Y, alphabet, multi_task, args,val_x=None,val_y=None):
    startDate = datetime.now()
    if args.track_name is not None:
        out = os.path.join(get_config().get("folder", "results_prefix"), startDate.strftime('%Y-%m-%d'),  args.track_name)
    else:
        out = os.path.join(get_config().get("folder", "results_prefix"), startDate.strftime('%Y-%m-%d'), startDate.strftime("%H-%M-%S"))
    a_size = len(alphabet) - 1 #alphabet contains 0 character for '', however we ignore that one
    dataset = make_dataset(X,Y,a_size,args.embed)
    # Make train, test and val set
    train, test, val = split_dataset(dataset)
    print len(train), len(test), len(val)
    logs = {'train':len(train), 'test':len(test), 'val':len(val)}
    # We make use of the multi process iteration such that the one-hot-encoding is done on a separate thread from the
    # GPU controller
    train_iter = I.MultiprocessIterator(train, args.batch_size)
    test_iter = I.MultiprocessIterator(test, args.batch_size, shuffle=False)
    val_iter = I.MultiprocessIterator(val, args.batch_size, shuffle=False, repeat=False)
    date = time.strftime("%Y-%m-%d_%H-%M-%S")

    fn_a = 'loss_%s.png' % date
    fn_a_veg = 'loss_%s_veg.png' % date
    fn_a_cus = 'loss_%s_cuisine.png' % date

    fn_b = 'lr_%s.png' % date
    fn_c = 'acc_%s.png' % date
    fn_c_veg = 'acc_%s_veg.png' % date
    fn_c_cus = 'acc_%s_cus.png' % date

    loss_r = E.PlotReport(['validation/main/loss', 'main/loss'], 'epoch', file_name=fn_a)
    loss_r_veg = E.PlotReport(['validation/main/loss/veg', 'main/loss/veg'], 'epoch', file_name=fn_a_veg)
    loss_r_cus = E.PlotReport(['validation/main/loss/cuisine', 'main/loss/cuisine'], 'epoch', file_name=fn_a_cus)
    lr_r = E.PlotReport(['lr'], 'epoch', file_name=fn_b)

    acc_r = E.PlotReport(['validation/main/accuracy','main/accuracy'], 'epoch', file_name=fn_c)
    acc_r_cus = E.PlotReport(['validation/main/accuracy/cuisine', 'main/accuracy/cuisine'], 'epoch', file_name=fn_c_cus)
    acc_r_veg = E.PlotReport(['validation/main/accuracy/veg', 'main/accuracy/veg'], 'epoch', file_name=fn_c_veg)

    if not multi_task:
        classifier = L.Classifier(model)
    else:
        tasks = [
            {'name': 'cuisine', 'factor': args.cus_factor, 'loss_fun': F.loss.softmax_cross_entropy.softmax_cross_entropy,
             'acc_fun': F.evaluation.accuracy.Accuracy(ignore_label=-1)},
            {'name': 'veg', 'factor': 1. - args.veg_factor, 'loss_fun': F.loss.softmax_cross_entropy.softmax_cross_entropy,
             'acc_fun': F.evaluation.accuracy.Accuracy(ignore_label=-1)}
        ]
        classifier = MultiTaskClassifier(model, tasks)
    classifier.compute_accuracy = args.acc_train # only probabilities for training
    if args.gpu >= 0:
        classifier.to_gpu(args.gpu)
    print "Model intialized, epochs: %d" % args.epochs
    optimizer = O.MomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(classifier)

    # prepare model for evaluation
    eval_model = classifier.copy()
    eval_model.compute_accuracy = args.acc_val
    eval_model.predictor.train = False

    updater = T.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = T.Trainer(updater, (args.epochs, 'epoch'), out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'
    if args.no_validation:
        trainer.extend(E.Evaluator(
            val_iter, eval_model, device=args.gpu))

    if args.val_ingredients is not None and args.val_chars is not None:
        X, Y, alphabet = get_dataset(args.val_chars, args.case_sensitive, args.ds_parts, max_recipe_size=args.val_ingredients,
                                     test_mode=args.test)
        dataset = make_dataset(X,Y,a_size,args.embed)
        # Make train, test and val set
        train_2, tes_2, val_2 = split_dataset(dataset)
        val2_iter = I.MultiprocessIterator(val_2, args.batch_size, shuffle=False, repeat=False)
        eval2_model = classifier.copy()
        eval2_model.compute_accuracy = args.acc_val
        eval2_model.predictor.train = False
        eval2_model.predictor.recipe_size = args.val_ingredients
        eval2_model.predictor.ingredient.width = args.val_chars
        ev2 = E.Evaluator(val2_iter, eval2_model, device=args.gpu)
        ev2.default_name = 'validation_2nd'
        trainer.extend(ev2)

        fn_d = 'acc_2nd_%s.png' % date
        acc_2nd = E.PlotReport(['validation_2nd/main/accuracy'], 'epoch', file_name=fn_d)
        trainer.extend(acc_2nd)

    trainer.extend(E.dump_graph('main/loss'))
    trainer.extend(E.ExponentialShift('lr', 0.5),
                   trigger=(7, 'epoch'))
    #trainer.extend(E.snapshot(), trigger=val_interval)
    if args.take_model_snapshot:
        trainer.extend(E.snapshot_object(
            model, 'model'), trigger=(args.epochs,'epoch'))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(E.LogReport(trigger=log_interval))
    trainer.extend(E.observe_lr(), trigger=log_interval)
    trainer.extend(E.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss','validation_2nd/main/loss' 'validation/main/accuracy/veg','validation/main/accuracy/cuisine', 'lr', 'main/loss/cuisine','main/loss/veg'
    ]), trigger=log_interval)
    trainer.extend(loss_r)
    trainer.extend(lr_r)
    trainer.extend(acc_r)
    trainer.extend(loss_r_veg)
    trainer.extend(loss_r_cus)
    trainer.extend(acc_r_veg)
    trainer.extend(acc_r_cus)

    trainer.extend(T.extensions.ProgressBar(update_interval=10))
    start = time.time()
    trainer.run()
    duration = time.time() - start
    re_s = len(train)*args.epochs/duration
    logs.update({'duration': duration, 'epochs': args.epochs, 'recipes': len(train), 'recipes/s': re_s, 'recipes/m': re_s*60})
    with open(os.path.join(out, 'log.json'), 'w') as io:
        json.dump(logs, io)



def train(args):
    n_chars = args.num_chars
    X,Y,alphabet = get_dataset(n_chars, args.case_sensitive, args.ds_parts, max_recipe_size=args.num_ingredients,test_mode=args.test)
    a_size = len(alphabet) -1
    print len(X),len([1 for y in Y if y == 1]),len([1 for y in Y if y == 0])
    print "Alphabet consists of %d characters" % a_size
    mt = args.ds_parts in ['intersect', 'union']


    model = build_model(a_size, 2, recipe_size=args.num_ingredients, recipe_type=args.recipe_type, ing_type=args.ingredient_type, num_chars=args.num_chars, multi_task=mt,embed=args.embed)
    start = time.time()
    train_model(model, X, Y, alphabet, mt, args)

def test_one_hot_encoding(X, alphabet):
    from dataset import print_recipe, print_recipe_oe
    dataset = OneHotEncodingDataset(X, len(alphabet)-1, dtype=np.int8)
    print_recipe(X[0], alphabet)
    print "-------"
    print_recipe_oe(dataset[0], alphabet)


if __name__ == "__main__":
    args = get_args()
    train(args)


