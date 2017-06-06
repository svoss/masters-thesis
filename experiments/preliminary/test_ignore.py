import chainer.functions as F
import numpy as np
if __name__ == "__main__":
    X = np.array(
        [[1.0,0.0,0.0],
        [1.0,0.0,0.0],
        [0.0, 1.0, 0.0]],
        np.float64)

    T = np.array([0,0,-1], np.int32)

    print F.loss.softmax_cross_entropy.softmax_cross_entropy(X,T).data