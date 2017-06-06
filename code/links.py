from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from function import MultiTaskLoss
from chainer import cuda
import chainer.functions as F
class MultiTaskClassifier(link.Chain):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor, tasks):
        super(MultiTaskClassifier, self).__init__(predictor=predictor)
        self.factors = [t['factor'] for t in tasks]
        self.tasks = tasks
        self.multitask = MultiTaskLoss(self.factors)


    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]

        self.y = None
        self.loss = None
        self.losses = []
        self.accuracy = None
        self.y = self.predictor(*x)
        xp = cuda.get_array_module(*x)

        reporter.report({'loss': self.loss}, self)
        Y = F.array.separate.separate(self.y,2)
        T = F.array.separate.separate(t, 1)
        for i,task in enumerate(self.tasks):
            y = Y[i]
            t = T[i]
            y.data = xp.ascontiguousarray(y.data)
            t.data = xp.ascontiguousarray(t.data)

            self.losses.append(task['loss_fun'](y,t, use_cudnn=True))
            reporter.report({'loss/' + task['name']: self.losses[i]}, self)
            if self.compute_accuracy:
                self.accuracy = task['acc_fun'](y, t)
                reporter.report({'accuracy/'+task['name']: self.accuracy}, self)


        self.losses = F.stack(self.losses)

        self.loss = self.multitask(self.losses)
        reporter.report({'loss':self.loss},self)
        return self.loss