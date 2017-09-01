import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class MultiTaskLoss(function.Function):

    """Padding of an array"""

    def __init__(self, factors):
        self.factors = factors
        self.factors_xp = None

    def check_type_forward(self, in_types):
        # Depending on the arguments, pad_width and keywords, the input value
        # may be inappropriate. In that case, numpy.pad or cupy.pad will raise
        # errors, so that only check the size and the dtype in this function.
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

        type_check.expect(x_type.shape[0] == len(self.factors))


    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        if self.factors_xp is None:
            self.factors_xp = xp.array(self.factors,dtype=inputs[0].dtype)
        x = inputs[0].dot(self.factors_xp)

        return xp.array(x,dtype=inputs[0].dtype).reshape(()),

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        gy = grads[0]
        gx = gy.dot(self.factors_xp).astype(x.dtype, copy=False)
        return gx,

if __name__ == "__main__":
    import numpy as np
    MTL = MultiTaskLoss([.33,.33,.33])
    L = np.array([[1.,1.,1.],[1.,2.,3.]],dtype=np.float64)
    X = MTL(L)
    print X.data
    X.backward()