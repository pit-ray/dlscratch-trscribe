import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0 import functions as F


def pattern1():
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()

    print(y)
    print(x.grad)


def pattern2():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()

    print(y)
    print(x.grad)


def pattern3():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()

    print(y)  # 5 7 9
    print(x.grad)

    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)


pattern1()
pattern2()
pattern3()
