import matplotlib
# This makes plots work via ssh
matplotlib.use('Agg')

import sys
from os.path import dirname,realpath
import os
path = os.path.join(dirname(dirname(realpath(__file__))), '../code')
sys.path.append(path)
import argparse
from dataset import print_recipe_oe
from commoncrawl_dataset import get_nutrition_dataset
from model import get_nutrition_model
import chainer.training.extensions as E
import chainer.iterators as I
import chainer.training as T
from config import get_config
from datetime import datetime
import time
import json
import chainer.optimizers as O
import chainer
def build_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr',type=float,default='.01')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--momentum',type=float,default=.0)
    parser.add_argument('--case-sensitive',action='store_true')
    parser.add_argument('--num-chars',type=int, default=96)
    parser.add_argument('--num-ingredients',type=int, default=24)
    parser.add_argument('--track-name',type=str,default=None)
    parser.add_argument('--take-model-snapshot',action='store_true')
    parser.add_argument('--dataset',type=str,default='calories')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--limit-to', type=int, default=None)
    parser.add_argument('--no-validation',action='store_false')
    parser.add_argument('--network',type=str)
    parser.add_argument('--categories',type=int,default=None)
    parser.add_argument('--load-model',type=str,default=None)
    parser.add_argument('--save-intermediate',type=int, default=None)
    parser.add_argument('--resume-from-iteration',type=int, default=None)
    return parser.parse_args()

def train_model(model, (train,val,test), args):
    if args.load_model is not None:
        if not os.path.exists(args.load_model):
            sys.exit("Path %s does not exist" % (args.load_model))

    startDate = datetime.now()
    if args.track_name is not None:
        out = os.path.join(get_config().get("folder", "results_prefix"), startDate.strftime('%Y-%m-%d'),  args.track_name)
    else:
        out = os.path.join(get_config().get("folder", "results_prefix"), startDate.strftime('%Y-%m-%d'), startDate.strftime("%H-%M-%S"))

    # Make train, test and val set
    logs = {'train':len(train), 'test':len(test), 'val':len(val)}
    # We make use of the multi process iteration such that the one-hot-encoding is done on a separate thread from the
    # GPU controller
    train_iter = I.SerialIterator(train, args.batch_size)

    #test_iter = I.MultiprocessIterator(test, args.batch_size, shuffle=False)
    val_iter = I.SerialIterator(val, args.batch_size, shuffle=False, repeat=False)

    fn_a = 'loss.png'
    fn_b = 'lr.png'
    fn_c = 'acc.png'
    lr_r = E.PlotReport(['lr'], 'epoch', file_name=fn_b)
    loss_r = E.PlotReport(['validation/main/loss', 'main/loss'], 'epoch', file_name=fn_a)
    acc_r = E.PlotReport(['validation/main/accuracy','main/accuracy'],'epoch',file_name=fn_c)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    print "Model intialized, epochs: %d" % args.epochs
    optimizer = O.MomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)

    # prepare model for evaluation
    eval_model = model.copy()
    eval_model.predictor.train = False

    updater = T.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = T.Trainer(updater, (args.epochs, 'epoch'), out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'
    if args.no_validation:
        trainer.extend(E.Evaluator(
            val_iter, eval_model, device=args.gpu))

    trainer.extend(E.dump_graph('main/loss'))
    trainer.extend(E.ExponentialShift('lr', 0.75), trigger=(4, 'epoch'))
    if args.save_intermediate is not None:
        print "saving intermediate results"
        trainer.extend(E.snapshot(), trigger=(args.save_intermediate,'epoch'))
    
    if args.take_model_snapshot:
        trainer.extend(E.snapshot_object(model, 'model'), trigger=(args.epochs,'epoch'))
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(E.LogReport(trigger=log_interval))
    trainer.extend(E.observe_lr(), trigger=log_interval)
    trainer.extend(E.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss','lr','main/accuracy','validation/main/accuracy'
    ]), trigger=log_interval)
    trainer.extend(loss_r)
    trainer.extend(lr_r)
    if args.no_validation:
        trainer.extend(acc_r)
    trainer.extend(T.extensions.ProgressBar(update_interval=10))
    
    if args.resume_from_iteration:
        fn = os.path.join(out,"snapshot_iter_%d" % args.resume_from_iteration)
        if not os.path.exists(fn):
            print "%s does not exist" % fn
            sys.exit()
        else:
            print "Resuming"
            chainer.serializers.load_npz(fn, trainer)
    
    start = time.time()
    trainer.run()
    duration = time.time() - start
    re_s = len(train)*args.epochs/duration
    logs.update({'duration': duration, 'epochs': args.epochs, 'recipes': len(train), 'recipes/s': re_s, 'recipes/m': re_s*60})
    with open(os.path.join(out, 'log.json'), 'w') as io:
        json.dump(logs, io)


if __name__ == "__main__":
    args = build_args()
    train, val, test, alphabet_size = get_nutrition_dataset(args)

    print "Training/validation/test size: %d/%d/%d " % (len(train), len(val), len(test))
    model = get_nutrition_model(args, alphabet_size)
    train_model(model, (train,val,test), args)

